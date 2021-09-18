# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple, List

import torch
import torch.nn.functional as F
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points, MonteCarloRaysampler, NDCGridRaysampler, RayBundle, EmissionAbsorptionRaymarcher
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.sample_pdf import sample_pdf
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
    _shifted_cumprod,
)
from .utils import calc_mse, calc_psnr, sample_images_at_mc_locs


class LinearWithRepeat(torch.nn.Linear):
    """
    if x has shape (..., k, n1)
    and y has shape (..., n2)
    then
    LinearWithRepeat(n1 + n2, out_features).forward((x,y))
    is equivalent to
    Linear(n1 + n2, out_features).forward(
        torch.cat([x, y.unsqueeze(-2).expand(..., k, n2)], dim=-1)
    )

    Or visually:
    Given the following, for each ray,

                feature   ->

    ray         xxxxxxxx
    position    xxxxxxxx
      |         xxxxxxxx
      v         xxxxxxxx


    and
                            yyyyyyyy

    where the y's do not depend on the position
    but only on the ray,
    we want to evaluate a Linear layer on both
    types of data at every position.

    It's as if we constructed

                xxxxxxxxyyyyyyyy
                xxxxxxxxyyyyyyyy
                xxxxxxxxyyyyyyyy
                xxxxxxxxyyyyyyyy

    and sent that through the Linear.
    """

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]:
            ```
            [
                sin(x[..., i]),
                sin(f_1*x[..., i]),
                sin(f_2*x[..., i]),
                ...
                sin(f_N * x[..., i]),
                cos(x[..., i]),
                cos(f_1*x[..., i]),
                cos(f_2*x[..., i]),
                ...
                cos(f_N * x[..., i]),
                x[..., i]     # only present if include_input is True.
            ]
            ```
        where N corresponds to `n_harmonic_functions`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.
        The shape of the output is [minibatch, ... , dim * (2 * N + 1)] if
        include_input is True, otherwise [minibatch, ... , dim * (2 * N)].

        If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
        powers of 2:
            `f_1 = 1, ..., f_N = 2**torch.arange(n_harmonic_functions)`

        If `logspace==False`, frequencies are linearly spaced between
        `1.0` and `2**(n_harmonic_functions-1)`:
            `f_1, ..., f_N = torch.linspace(
                1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
            )`

        Note that `x` is also premultiplied by the base frequency `omega0`
        before evaluating the harmonic functions.
        """
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        try:
            self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        except TypeError:
            # workaround for pytorch<1.6
            self.register_buffer("_frequencies", omega0 * frequencies)

        self.include_input = include_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x` of shape
                [..., dim * (n_harmonic_functions * 2 + T)] where
                T is 1 if include_input is True and 0 otherwise.
        """
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)
        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)


class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions_xyz: int = 6,
        n_harmonic_functions_dir: int = 4,
        n_hidden_neurons_xyz: int = 256,
        n_hidden_neurons_dir: int = 128,
        n_layers_xyz: int = 8,
        append_xyz: Tuple[int] = (5,),
        use_multiple_streams: bool = True,
        **kwargs,
    ):
        """
        Args:
            n_harmonic_functions_xyz: The number of harmonic functions
                used to form the harmonic embedding of 3D point locations.
            n_harmonic_functions_dir: The number of harmonic functions
                used to form the harmonic embedding of the ray directions.
            n_hidden_neurons_xyz: The number of hidden units in the
                fully connected layers of the MLP that accepts the 3D point
                locations and outputs the occupancy field with the intermediate
                features.
            n_hidden_neurons_dir: The number of hidden units in the
                fully connected layers of the MLP that accepts the intermediate
                features and ray directions and outputs the radiance field
                (per-point colors).
            n_layers_xyz: The number of layers of the MLP that outputs the
                occupancy field.
            append_xyz: The list of indices of the skip layers of the occupancy MLP.
            use_multiple_streams: Whether density and color should be calculated on
                separate CUDA streams.
        """
        super().__init__()

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

        self.mlp_xyz = MLPWithInputSkips(
            n_layers_xyz,
            embedding_dim_xyz,
            n_hidden_neurons_xyz,
            embedding_dim_xyz,
            n_hidden_neurons_xyz,
            input_skips=append_xyz,
        )

        self.intermediate_linear = torch.nn.Linear(
            n_hidden_neurons_xyz, n_hidden_neurons_xyz
        )
        _xavier_init(self.intermediate_linear)

        self.density_layer = torch.nn.Linear(n_hidden_neurons_xyz, 1)
        _xavier_init(self.density_layer)

        # Zero the bias of the density layer to avoid
        # a completely transparent initialization.
        self.density_layer.bias.data[:] = 0.0  # fixme: Sometimes this is not enough

        self.color_layer = torch.nn.Sequential(
            LinearWithRepeat(
                n_hidden_neurons_xyz + embedding_dim_dir, n_hidden_neurons_dir
            ),
            torch.nn.ReLU(True),
            torch.nn.Linear(n_hidden_neurons_dir, 3),
            torch.nn.Sigmoid(),
        )
        self.use_multiple_streams = use_multiple_streams

    def _get_densities(
        self,
        features: torch.Tensor,
        depth_values: torch.Tensor,
        density_noise_std: float,
    ) -> torch.Tensor:
        """
        This function takes `features` predicted by `self.mlp_xyz`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later re-weighted using the depth step sizes
        and mapped to [0-1] range with 1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        )[..., None]
        if density_noise_std > 0.0:
            raw_densities = (
                raw_densities + torch.randn_like(raw_densities) * density_noise_std
            )
        densities = 1 - (-deltas * torch.relu(raw_densities)).exp()
        return densities

    def _get_colors(
        self, features: torch.Tensor, rays_directions: torch.Tensor
    ) -> torch.Tensor:
        """
        This function takes per-point `features` predicted by `self.mlp_xyz`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(rays_directions, dim=-1)

        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        return self.color_layer((self.intermediate_linear(features), rays_embedding))

    def _get_densities_and_colors(
        self, features: torch.Tensor, ray_bundle: RayBundle, density_noise_std: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The second part of the forward calculation.

        Args:
            features: the output of the common mlp (the prior part of the
                calculation), shape
                (minibatch x ... x self.n_hidden_neurons_xyz).
            ray_bundle: As for forward().
            density_noise_std:  As for forward().

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        if self.use_multiple_streams and features.is_cuda:
            current_stream = torch.cuda.current_stream(features.device)
            other_stream = torch.cuda.Stream(features.device)
            other_stream.wait_stream(current_stream)

            with torch.cuda.stream(other_stream):
                rays_densities = self._get_densities(
                    features, ray_bundle.lengths, density_noise_std
                )
                # rays_densities.shape = [minibatch x ... x 1] in [0-1]

            rays_colors = self._get_colors(features, ray_bundle.directions)
            # rays_colors.shape = [minibatch x ... x 3] in [0-1]

            current_stream.wait_stream(other_stream)
        else:
            # Same calculation as above, just serial.
            rays_densities = self._get_densities(
                features, ray_bundle.lengths, density_noise_std
            )
            rays_colors = self._get_colors(features, ray_bundle.directions)
        return rays_densities, rays_colors

    def forward(
        self,
        ray_bundle: RayBundle,
        density_noise_std: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            density_noise_std: A floating point value representing the
                variance of the random normal noise added to the output of
                the opacity function. This can prevent floating artifacts.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions*6 + 3]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(embeds_xyz, embeds_xyz)
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

        rays_densities, rays_colors = self._get_densities_and_colors(
            features, ray_bundle, density_noise_std
        )
        return rays_densities, rays_colors


class MLPWithInputSkips(torch.nn.Module):
    """
    Implements the multi-layer perceptron architecture of the Neural Radiance Field.

    As such, `MLPWithInputSkips` is a multi layer perceptron consisting
    of a sequence of linear layers with ReLU activations.

    Additionally, for a set of predefined layers `input_skips`, the forward pass
    appends a skip tensor `z` to the output of the preceding layer.

    Note that this follows the architecture described in the Supplementary
    Material (Fig. 7) of [1].

    References:
        [1] Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik
            and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng:
            NeRF: Representing Scenes as Neural Radiance Fields for View
            Synthesis, ECCV2020
    """

    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips: Tuple[int] = (),
    ):
        """
        Args:
            n_layers: The number of linear layers of the MLP.
            input_dim: The number of channels of the input tensor.
            output_dim: The number of channels of the output.
            skip_dim: The number of channels of the tensor `z` appended when
                evaluating the skip layers.
            hidden_dim: The number of hidden units of the MLP.
            input_skips: The list of layer indices at which we append the skip
                tensor `z`.
        """
        super().__init__()
        layers = []
        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim
            linear = torch.nn.Linear(dimin, dimout)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The input tensor of shape `(..., input_dim)`.
            z: The input skip tensor of shape `(..., skip_dim)` which is appended
                to layers whose indices are specified by `input_skips`.
        Returns:
            y: The output tensor of shape `(..., output_dim)`.
        """
        y = x
        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)
            y = layer(y)
        return y


class EmissionAbsorptionNeRFRaymarcher(EmissionAbsorptionRaymarcher):
    """
    This is essentially the `pytorch3d.renderer.EmissionAbsorptionRaymarcher`
    which additionally returns the rendering weights. It also skips returning
    the computation of the alpha-mask which is, in case of NeRF, equal to 1
    everywhere.

    The weights are later used in the NeRF pipeline to carry out the importance
    ray-sampling for the fine rendering pass.

    For more details about the EmissionAbsorptionRaymarcher please refer to
    the documentation of `pytorch3d.renderer.EmissionAbsorptionRaymarcher`.
    """

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        eps: float = 1e-10,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)` whose values range in [0, 1].
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            eps: A lower bound added to `rays_densities` before computing
                the absorption function (cumprod of `1-rays_densities` along
                each ray). This prevents the cumprod to yield exact 0
                which would inhibit any gradient-based learning.

        Returns:
            features: A tensor of shape `(..., feature_dim)` containing
                the rendered features for each ray.
            weights: A tensor of shape `(..., n_points_per_ray)` containing
                the ray-specific emission-absorption distribution.
                Each ray distribution `(..., :)` is a valid probability
                distribution, i.e. it contains non-negative values that integrate
                to 1, such that `weights.sum(dim=-1)==1).all()` yields `True`.
        """
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            None,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )
        _check_density_bounds(rays_densities)
        rays_densities = rays_densities[..., 0]
        absorption = _shifted_cumprod(
            (1.0 + eps) - rays_densities, shift=self.surface_thickness
        )
        weights = rays_densities * absorption
        features = (weights[..., None] * rays_features).sum(dim=-2)

        return features, weights


class ProbabilisticRaysampler(torch.nn.Module):
    """
    Implements the importance sampling of points along rays.
    The input is a `RayBundle` object with a `ray_weights` tensor
    which specifies the probabilities of sampling a point along each ray.

    This raysampler is used for the fine rendering pass of NeRF.
    As such, the forward pass accepts the RayBundle output by the
    raysampling of the coarse rendering pass. Hence, it does not
    take cameras as input.
    """

    def __init__(
        self,
        n_pts_per_ray: int,
        stratified: bool,
        stratified_test: bool,
        add_input_samples: bool = True,
    ):
        """
        Args:
            n_pts_per_ray: The number of points to sample along each ray.
            stratified: If `True`, the input `ray_weights` are assumed to be
                sampled at equidistant intervals.
            stratified_test: Same as `stratified` with the difference that this
                setting is applied when the module is in the `eval` mode
                (`self.training==False`).
            add_input_samples: Concatenates and returns the sampled values
                together with the input samples.
        """
        super().__init__()
        self._n_pts_per_ray = n_pts_per_ray
        self._stratified = stratified
        self._stratified_test = stratified_test
        self._add_input_samples = add_input_samples

    def forward(
        self,
        input_ray_bundle: RayBundle,
        ray_weights: torch.Tensor,
        **kwargs,
    ) -> RayBundle:
        """
        Args:
            input_ray_bundle: An instance of `RayBundle` specifying the
                source rays for sampling of the probability distribution.
            ray_weights: A tensor of shape
                `(..., input_ray_bundle.legths.shape[-1])` with non-negative
                elements defining the probability distribution to sample
                ray points from.

        Returns:
            ray_bundle: A new `RayBundle` instance containing the input ray
                points together with `n_pts_per_ray` additional sampled
                points per ray.
        """

        # Calculate the mid-points between the ray depths.
        z_vals = input_ray_bundle.lengths
        batch_size = z_vals.shape[0]

        # Carry out the importance sampling.
        with torch.no_grad():
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid.view(-1, z_vals_mid.shape[-1]),
                ray_weights.view(-1, ray_weights.shape[-1])[..., 1:-1],
                self._n_pts_per_ray,
                det=not (
                    (self._stratified and self.training)
                    or (self._stratified_test and not self.training)
                ),
            ).view(batch_size, z_vals.shape[1], self._n_pts_per_ray)

        if self._add_input_samples:
            # Add the new samples to the input ones.
            z_vals = torch.cat((z_vals, z_samples), dim=-1)
        else:
            z_vals = z_samples
        # Resort by depth.
        z_vals, _ = torch.sort(z_vals, dim=-1)

        return RayBundle(
            origins=input_ray_bundle.origins,
            directions=input_ray_bundle.directions,
            lengths=z_vals,
            xys=input_ray_bundle.xys,
        )


class NeRFRaysampler(torch.nn.Module):
    """
    Implements the raysampler of NeRF.

    Depending on the `self.training` flag, the raysampler either samples
    a chunk of random rays (`self.training==True`), or returns a subset of rays
    of the full image grid (`self.training==False`).
    The chunking of rays allows for efficient evaluation of the NeRF implicit
    surface function without encountering out-of-GPU-memory errors.

    Additionally, this raysampler supports pre-caching of the ray bundles
    for a set of input cameras (`self.precache_rays`).
    Pre-caching the rays before training greatly speeds-up the ensuing
    raysampling step of the training NeRF iterations.
    """

    def __init__(
        self,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        n_rays_per_image: int,
        image_width: int,
        image_height: int,
        stratified: bool = False,
        stratified_test: bool = False,
    ):
        """
        Args:
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of a ray-point.
            max_depth: The maximum depth of a ray-point.
            n_rays_per_image: Number of Monte Carlo ray samples when training
                (`self.training==True`).
            image_width: The horizontal size of the image grid.
            image_height: The vertical size of the image grid.
            stratified: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during training (`self.training==True`).
            stratified_test: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during evaluation (`self.training==False`).
        """

        super().__init__()
        self._stratified = stratified
        self._stratified_test = stratified_test

        # Initialize the grid ray sampler.
        self._grid_raysampler = NDCGridRaysampler(
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        # Initialize the Monte Carlo ray sampler.
        self._mc_raysampler = MonteCarloRaysampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            n_rays_per_image=n_rays_per_image,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        # create empty ray cache
        self._ray_cache = {}

    def get_n_chunks(self, chunksize: int, batch_size: int):
        """
        Returns the total number of `chunksize`-sized chunks
        of the raysampler's rays.

        Args:
            chunksize: The number of rays per chunk.
            batch_size: The size of the batch of the raysampler.

        Returns:
            n_chunks: The total number of chunks.
        """
        return int(
            math.ceil(
                (self._grid_raysampler._xy_grid.numel() * 0.5 * batch_size) / chunksize
            )
        )

    def _print_precaching_progress(self, i, total, bar_len=30):
        """
        Print a progress bar for ray precaching.
        """
        position = round((i + 1) / total * bar_len)
        pbar = "[" + "█" * position + " " * (bar_len - position) + "]"
        print(pbar, end="\r")

    def precache_rays(self, cameras: List[CamerasBase], camera_hashes: List):
        """
        Precaches the rays emitted from the list of cameras `cameras`,
        where each camera is uniquely identified with the corresponding hash
        from `camera_hashes`.

        The cached rays are moved to cpu and stored in `self._ray_cache`.
        Raises `ValueError` when caching two cameras with the same hash.

        Args:
            cameras: A list of `N` cameras for which the rays are pre-cached.
            camera_hashes: A list of `N` unique identifiers of each
                camera from `cameras`.
        """
        print(f"Precaching {len(cameras)} ray bundles ...")
        full_chunksize = (
            self._grid_raysampler._xy_grid.numel()
            // 2
            * self._grid_raysampler._n_pts_per_ray
        )
        if self.get_n_chunks(full_chunksize, 1) != 1:
            raise ValueError("There has to be one chunk for precaching rays!")
        for camera_i, (camera, camera_hash) in enumerate(zip(cameras, camera_hashes)):
            ray_bundle = self.forward(
                camera,
                caching=True,
                chunksize=full_chunksize,
            )
            if camera_hash in self._ray_cache:
                raise ValueError("There are redundant cameras!")
            self._ray_cache[camera_hash] = RayBundle(
                *[v.to("cpu").detach() for v in ray_bundle]
            )
            self._print_precaching_progress(camera_i, len(cameras))
        print("")

    def _stratify_ray_bundle(self, ray_bundle: RayBundle):
        """
        Stratifies the lengths of the input `ray_bundle`.

        More specifically, the stratification replaces each ray points' depth `z`
        with a sample from a uniform random distribution on
        `[z - delta_depth, z+delta_depth]`, where `delta_depth` is the difference
        of depths of the consecutive ray depth values.

        Args:
            `ray_bundle`: The input `RayBundle`.

        Returns:
            `stratified_ray_bundle`: `ray_bundle` whose `lengths` field is replaced
                with the stratified samples.
        """
        z_vals = ray_bundle.lengths
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        z_vals = lower + (upper - lower) * torch.rand_like(lower)
        return ray_bundle._replace(lengths=z_vals)

    def _normalize_raybundle(self, ray_bundle: RayBundle):
        """
        Normalizes the ray directions of the input `RayBundle` to unit norm.
        """
        ray_bundle = ray_bundle._replace(
            directions=torch.nn.functional.normalize(ray_bundle.directions, dim=-1)
        )
        return ray_bundle

    def forward(
        self,
        cameras: CamerasBase,
        chunksize: int = None,
        chunk_idx: int = 0,
        camera_hash: str = None,
        caching: bool = False,
        **kwargs,
    ) -> RayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            chunksize: The number of rays per chunk.
                Active only when `self.training==False`.
            chunk_idx: The index of the ray chunk. The number has to be in
                `[0, self.get_n_chunks(chunksize, batch_size)-1]`.
                Active only when `self.training==False`.
            camera_hash: A unique identifier of a pre-cached camera. If `None`,
                the cache is not searched and the rays are calculated from scratch.
            caching: If `True`, activates the caching mode that returns the `RayBundle`
                that should be stored into the cache.
        Returns:
            A named tuple `RayBundle` with the following fields:
                origins: A tensor of shape
                    `(batch_size, n_rays_per_image, 3)`
                    denoting the locations of ray origins in the world coordinates.
                directions: A tensor of shape
                    `(batch_size, n_rays_per_image, 3)`
                    denoting the directions of each ray in the world coordinates.
                lengths: A tensor of shape
                    `(batch_size, n_rays_per_image, n_pts_per_ray)`
                    containing the z-coordinate (=depth) of each ray in world units.
                xys: A tensor of shape
                    `(batch_size, n_rays_per_image, 2)`
                    containing the 2D image coordinates of each ray.
        """

        batch_size = cameras.R.shape[0]  # pyre-ignore
        device = cameras.device

        if (camera_hash is None) and (not caching) and self.training:
            # Sample random rays from scratch.
            ray_bundle = self._mc_raysampler(cameras)
            ray_bundle = self._normalize_raybundle(ray_bundle)
        else:
            if camera_hash is not None:
                # The case where we retrieve a camera from cache.
                if batch_size != 1:
                    raise NotImplementedError(
                        "Ray caching works only for batches with a single camera!"
                    )
                full_ray_bundle = self._ray_cache[camera_hash]
            else:
                # We generate a full ray grid from scratch.
                full_ray_bundle = self._grid_raysampler(cameras)
                full_ray_bundle = self._normalize_raybundle(full_ray_bundle)

            n_pixels = full_ray_bundle.directions.shape[:-1].numel()

            if self.training:
                # During training we randomly subsample rays.
                sel_rays = torch.randperm(n_pixels, device=device)[
                    : self._mc_raysampler._n_rays_per_image
                ]
            else:
                # In case we test, we take only the requested chunk.
                if chunksize is None:
                    chunksize = n_pixels * batch_size
                start = chunk_idx * chunksize * batch_size
                end = min(start + chunksize, n_pixels)
                sel_rays = torch.arange(
                    start,
                    end,
                    dtype=torch.long,
                    device=full_ray_bundle.lengths.device,
                )

            # Take the "sel_rays" rays from the full ray bundle.
            ray_bundle = RayBundle(
                *[
                    v.view(n_pixels, -1)[sel_rays]
                    .view(batch_size, sel_rays.numel() // batch_size, -1)
                    .to(device)
                    for v in full_ray_bundle
                ]
            )

        if (
            (self._stratified and self.training)
            or (self._stratified_test and not self.training)
        ) and not caching:  # Make sure not to stratify when caching!
            ray_bundle = self._stratify_ray_bundle(ray_bundle)

        return ray_bundle


class RadianceFieldRenderer(torch.nn.Module):
    """
    Implements a renderer of a Neural Radiance Field.

    This class holds pointers to the fine and coarse renderer objects, which are
    instances of `pytorch3d.renderer.ImplicitRenderer`, and pointers to the
    neural networks representing the fine and coarse Neural Radiance Fields,
    which are instances of `NeuralRadianceField`.

    The rendering forward pass proceeds as follows:
        1) For a given input camera, rendering rays are generated with the
            `NeRFRaysampler` object of `self._renderer['coarse']`.
            In the training mode (`self.training==True`), the rays are a set
                of `n_rays_per_image` random 2D locations of the image grid.
            In the evaluation mode (`self.training==False`), the rays correspond
                to the full image grid. The rays are further split to
                `chunk_size_test`-sized chunks to prevent out-of-memory errors.
        2) For each ray point, the coarse `NeuralRadianceField` MLP is evaluated.
            The pointer to this MLP is stored in `self._implicit_function['coarse']`
        3) The coarse radiance field is rendered with the
            `EmissionAbsorptionNeRFRaymarcher` object of `self._renderer['coarse']`.
        4) The coarse raymarcher outputs a probability distribution that guides
            the importance raysampling of the fine rendering pass. The
            `ProbabilisticRaysampler` stored in `self._renderer['fine'].raysampler`
            implements the importance ray-sampling.
        5) Similar to 2) the fine MLP in `self._implicit_function['fine']`
            labels the ray points with occupancies and colors.
        6) self._renderer['fine'].raymarcher` generates the final fine render.
        7) The fine and coarse renders are compared to the ground truth input image
            with PSNR and MSE metrics.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        n_pts_per_ray: int,
        n_pts_per_ray_fine: int,
        n_rays_per_image: int,
        min_depth: float,
        max_depth: float,
        stratified: bool,
        stratified_test: bool,
        chunk_size_test: int,
        n_harmonic_functions_xyz: int = 6,
        n_harmonic_functions_dir: int = 4,
        n_hidden_neurons_xyz: int = 256,
        n_hidden_neurons_dir: int = 128,
        n_layers_xyz: int = 8,
        append_xyz: Tuple[int] = (5,),
        density_noise_std: float = 0.0,
        visualization: bool = False,
    ):
        """
        Args:
            image_size: The size of the rendered image (`[height, width]`).
            n_pts_per_ray: The number of points sampled along each ray for the
                coarse rendering pass.
            n_pts_per_ray_fine: The number of points sampled along each ray for the
                fine rendering pass.
            n_rays_per_image: Number of Monte Carlo ray samples when training
                (`self.training==True`).
            min_depth: The minimum depth of a sampled ray-point for the coarse rendering.
            max_depth: The maximum depth of a sampled ray-point for the coarse rendering.
            stratified: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during training (`self.training==True`).
            stratified_test: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during evaluation (`self.training==False`).
            chunk_size_test: The number of rays in each chunk of image rays.
                Active only when `self.training==True`.
            n_harmonic_functions_xyz: The number of harmonic functions
                used to form the harmonic embedding of 3D point locations.
            n_harmonic_functions_dir: The number of harmonic functions
                used to form the harmonic embedding of the ray directions.
            n_hidden_neurons_xyz: The number of hidden units in the
                fully connected layers of the MLP that accepts the 3D point
                locations and outputs the occupancy field with the intermediate
                features.
            n_hidden_neurons_dir: The number of hidden units in the
                fully connected layers of the MLP that accepts the intermediate
                features and ray directions and outputs the radiance field
                (per-point colors).
            n_layers_xyz: The number of layers of the MLP that outputs the
                occupancy field.
            append_xyz: The list of indices of the skip layers of the occupancy MLP.
                Prior to evaluating the skip layers, the tensor which was input to MLP
                is appended to the skip layer input.
            density_noise_std: The standard deviation of the random normal noise
                added to the output of the occupancy MLP.
                Active only when `self.training==True`.
            visualization: whether to store extra output for visualization.
        """

        super().__init__()

        # The renderers and implicit functions are stored under the fine/coarse
        # keys in ModuleDict PyTorch modules.
        self._renderer = torch.nn.ModuleDict()
        self._implicit_function = torch.nn.ModuleDict()

        # Init the EA raymarcher used by both passes.
        raymarcher = EmissionAbsorptionNeRFRaymarcher()

        # Parse out image dimensions.
        image_height, image_width = image_size

        for render_pass in ("coarse", "fine"):
            if render_pass == "coarse":
                # Initialize the coarse raysampler.
                raysampler = NeRFRaysampler(
                    n_pts_per_ray=n_pts_per_ray,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    stratified=stratified,
                    stratified_test=stratified_test,
                    n_rays_per_image=n_rays_per_image,
                    image_height=image_height,
                    image_width=image_width,
                )
            elif render_pass == "fine":
                # Initialize the fine raysampler.
                raysampler = ProbabilisticRaysampler(
                    n_pts_per_ray=n_pts_per_ray_fine,
                    stratified=stratified,
                    stratified_test=stratified_test,
                )
            else:
                raise ValueError(f"No such rendering pass {render_pass}")

            # Initialize the fine/coarse renderer.
            self._renderer[render_pass] = ImplicitRenderer(
                raysampler=raysampler,
                raymarcher=raymarcher,
            )

            # Instantiate the fine/coarse NeuralRadianceField module.
            self._implicit_function[render_pass] = NeuralRadianceField(
                n_harmonic_functions_xyz=n_harmonic_functions_xyz,
                n_harmonic_functions_dir=n_harmonic_functions_dir,
                n_hidden_neurons_xyz=n_hidden_neurons_xyz,
                n_hidden_neurons_dir=n_hidden_neurons_dir,
                n_layers_xyz=n_layers_xyz,
                append_xyz=append_xyz,
            )

        self._density_noise_std = density_noise_std
        self._chunk_size_test = chunk_size_test
        self._image_size = image_size
        self.visualization = visualization

    def precache_rays(
        self,
        cache_cameras: List[CamerasBase],
        cache_camera_hashes: List[str],
    ):
        """
        Precaches the rays emitted from the list of cameras `cache_cameras`,
        where each camera is uniquely identified with the corresponding hash
        from `cache_camera_hashes`.

        The cached rays are moved to cpu and stored in
        `self._renderer['coarse']._ray_cache`.

        Raises `ValueError` when caching two cameras with the same hash.

        Args:
            cache_cameras: A list of `N` cameras for which the rays are pre-cached.
            cache_camera_hashes: A list of `N` unique identifiers for each
                camera from `cameras`.
        """
        self._renderer["coarse"].raysampler.precache_rays(
            cache_cameras,
            cache_camera_hashes,
        )

    def _process_ray_chunk(
        self,
        camera_hash: Optional[str],
        camera: CamerasBase,
        image: torch.Tensor,
        chunk_idx: int,
    ) -> dict:
        """
        Samples and renders a chunk of rays.

        Args:
            camera_hash: A unique identifier of a pre-cached camera.
                If `None`, the cache is not searched and the sampled rays are
                calculated from scratch.
            camera: A batch of cameras from which the scene is rendered.
            image: A batch of corresponding ground truth images of shape
                ('batch_size', ·, ·, 3).
            chunk_idx: The index of the currently rendered ray chunk.
        Returns:
            out: `dict` containing the outputs of the rendering:
                `rgb_coarse`: The result of the coarse rendering pass.
                `rgb_fine`: The result of the fine rendering pass.
                `rgb_gt`: The corresponding ground-truth RGB values.
        """
        # Initialize the outputs of the coarse rendering to None.
        coarse_ray_bundle = None
        coarse_weights = None

        # First evaluate the coarse rendering pass, then the fine one.
        for renderer_pass in ("coarse", "fine"):
            (rgb, weights), ray_bundle_out = self._renderer[renderer_pass](
                cameras=camera,
                volumetric_function=self._implicit_function[renderer_pass],
                chunksize=self._chunk_size_test,
                chunk_idx=chunk_idx,
                density_noise_std=(self._density_noise_std if self.training else 0.0),
                input_ray_bundle=coarse_ray_bundle,
                ray_weights=coarse_weights,
                camera_hash=camera_hash,
            )

            if renderer_pass == "coarse":
                rgb_coarse = rgb
                # Store the weights and the rays of the first rendering pass
                # for the ensuing importance ray-sampling of the fine render.
                coarse_ray_bundle = ray_bundle_out
                coarse_weights = weights
                if image is not None:
                    # Sample the ground truth images at the xy locations of the
                    # rendering ray pixels.
                    rgb_gt = sample_images_at_mc_locs(
                        image[..., :3][None],
                        ray_bundle_out.xys,
                    )
                else:
                    rgb_gt = None

            elif renderer_pass == "fine":
                rgb_fine = rgb

            else:
                raise ValueError(f"No such rendering pass {renderer_pass}")

        out = {"rgb_fine": rgb_fine, "rgb_coarse": rgb_coarse, "rgb_gt": rgb_gt}
        if self.visualization:
            # Store the coarse rays/weights only for visualization purposes.
            out["coarse_ray_bundle"] = type(coarse_ray_bundle)(
                *[v.detach().cpu() for k, v in coarse_ray_bundle._asdict().items()]
            )
            out["coarse_weights"] = coarse_weights.detach().cpu()

        return out

    def forward(
        self,
        camera_hash: Optional[str],
        camera: CamerasBase,
        image: torch.Tensor,
    ) -> Tuple[dict, dict]:
        """
        Performs the coarse and fine rendering passes of the radiance field
        from the viewpoint of the input `camera`.
        Afterwards, both renders are compared to the input ground truth `image`
        by evaluating the peak signal-to-noise ratio and the mean-squared error.

        The rendering result depends on the `self.training` flag:
            - In the training mode (`self.training==True`), the function renders
              a random subset of image rays (Monte Carlo rendering).
            - In evaluation mode (`self.training==False`), the function renders
              the full image. In order to prevent out-of-memory errors,
              when `self.training==False`, the rays are sampled and rendered
              in batches of size `chunksize`.

        Args:
            camera_hash: A unique identifier of a pre-cached camera.
                If `None`, the cache is not searched and the sampled rays are
                calculated from scratch.
            camera: A batch of cameras from which the scene is rendered.
            image: A batch of corresponding ground truth images of shape
                ('batch_size', ·, ·, 3).
        Returns:
            out: `dict` containing the outputs of the rendering:
                `rgb_coarse`: The result of the coarse rendering pass.
                `rgb_fine`: The result of the fine rendering pass.
                `rgb_gt`: The corresponding ground-truth RGB values.

                The shape of `rgb_coarse`, `rgb_fine`, `rgb_gt` depends on the
                `self.training` flag:
                    If `==True`, all 3 tensors are of shape
                    `(batch_size, n_rays_per_image, 3)` and contain the result
                    of the Monte Carlo training rendering pass.
                    If `==False`, all 3 tensors are of shape
                    `(batch_size, image_size[0], image_size[1], 3)` and contain
                    the result of the full image rendering pass.
            metrics: `dict` containing the error metrics comparing the fine and
                coarse renders to the ground truth:
                `mse_coarse`: Mean-squared error between the coarse render and
                    the input `image`
                `mse_fine`: Mean-squared error between the fine render and
                    the input `image`
                `psnr_coarse`: Peak signal-to-noise ratio between the coarse render and
                    the input `image`
                `psnr_fine`: Peak signal-to-noise ratio between the fine render and
                    the input `image`
        """
        if not self.training:
            # Full evaluation pass.
            n_chunks = self._renderer["coarse"].raysampler.get_n_chunks(
                self._chunk_size_test,
                camera.R.shape[0],
            )
        else:
            # MonteCarlo ray sampling.
            n_chunks = 1

        # Process the chunks of rays.
        chunk_outputs = [
            self._process_ray_chunk(
                camera_hash,
                camera,
                image,
                chunk_idx,
            )
            for chunk_idx in range(n_chunks)
        ]

        if not self.training:
            # For a full render pass concatenate the output chunks,
            # and reshape to image size.
            out = {
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, *self._image_size, 3)
                if chunk_outputs[0][k] is not None
                else None
                for k in ("rgb_fine", "rgb_coarse", "rgb_gt")
            }
        else:
            out = chunk_outputs[0]

        # Calc the error metrics.
        metrics = {}
        if image is not None:
            for render_pass in ("coarse", "fine"):
                for metric_name, metric_fun in zip(
                    ("mse", "psnr"), (calc_mse, calc_psnr)
                ):
                    metrics[f"{metric_name}_{render_pass}"] = metric_fun(
                        out["rgb_" + render_pass][..., :3],
                        out["rgb_gt"][..., :3],
                    )

        return out, metrics


def visualize_nerf_outputs(
    nerf_out: dict, output_cache: List, viz: Visdom, visdom_env: str
):
    """
    Visualizes the outputs of the `RadianceFieldRenderer`.

    Args:
        nerf_out: An output of the validation rendering pass.
        output_cache: A list with outputs of several training render passes.
        viz: A visdom connection object.
        visdom_env: The name of visdom environment for visualization.
    """

    # Show the training images.
    ims = torch.stack([o["image"] for o in output_cache])
    ims = torch.cat(list(ims), dim=1)
    viz.image(
        ims.permute(2, 0, 1),
        env=visdom_env,
        win="images",
        opts={"title": "train_images"},
    )

    # Show the coarse and fine renders together with the ground truth images.
    ims_full = torch.cat(
        [
            nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
            for imvar in ("rgb_coarse", "rgb_fine", "rgb_gt")
        ],
        dim=2,
    )
    viz.image(
        ims_full,
        env=visdom_env,
        win="images_full",
        opts={"title": "coarse | fine | target"},
    )

    # Make a 3D plot of training cameras and their emitted rays.
    camera_trace = {
        f"camera_{ci:03d}": o["camera"].cpu() for ci, o in enumerate(output_cache)
    }
    ray_pts_trace = {
        f"ray_pts_{ci:03d}": Pointclouds(
            ray_bundle_to_ray_points(o["coarse_ray_bundle"])
            .detach()
            .cpu()
            .view(1, -1, 3)
        )
        for ci, o in enumerate(output_cache)
    }
    plotly_plot = plot_scene(
        {
            "training_scene": {
                **camera_trace,
                **ray_pts_trace,
            },
        },
        pointcloud_max_points=5000,
        pointcloud_marker_size=1,
        camera_scale=0.3,
    )
    viz.plotlyplot(plotly_plot, env=visdom_env, win="scenes")