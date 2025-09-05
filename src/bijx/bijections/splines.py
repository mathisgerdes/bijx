"""
Spline-based bijective transformations for normalizing flows.

This module provides rational quadratic spline transformations that offer
flexible, smooth bijections with trainable parameters.

Reference:
    Durkan et al. "Neural Spline Flows" (arXiv:1906.04032)
"""

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax_autovmap import autovmap

from ..utils import ShapeInfo
from .base import ApplyBijection


@autovmap(inputs=0, bin_widths=1, bin_heights=1, knot_slopes=1)
def rational_quadratic_spline(
    inputs,
    bin_widths,
    bin_heights,
    knot_slopes,
    *,
    inverse=False,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_slope=1e-3,
):
    r"""Apply monotonic rational quadratic spline transformation.

    Implements the rational quadratic spline bijection from Durkan et al.
    (arXiv:1906.04032). The transformation constructs a smooth, monotonic
    function using piecewise rational quadratic segments between knot points.

    Type: [0, 1] → [0, 1] (with identity extension outside domain)

    where $x' = (x - x_k)/w_k$ is the normalized position within bin $k$,
    and $s_k = h_k/w_k$ is the bin slope.

    Key features:
        - Monotonic by construction through parameter normalization
        - Identity transformation outside [0,1] domain
        - Numerically stable with minimum parameter constraints
        - Efficient inverse computation via quadratic formula

    Args:
        inputs: Input values to transform.
        bin_widths: Unnormalized bin widths (softmax applied internally).
        bin_heights: Unnormalized bin heights (softmax applied internally).
        knot_slopes: Internal knot slopes (softplus applied for positivity).
        inverse: Whether to apply inverse transformation.
        min_bin_width: Minimum bin width for numerical stability.
        min_bin_height: Minimum bin height for numerical stability.
        min_slope: Minimum knot slope for numerical stability.

    Returns:
        Tuple of (transformed_inputs, log_determinant) where log_determinant
        gives the log absolute Jacobian determinant of the transformation.

    Note:
        Boundary knot slopes are fixed to 1.0 to ensure smooth linear tails
        outside the spline domain. Only internal knot slopes are trainable.

    Example:
        >>> x = jnp.array([0.2, 0.5, 0.8])
        >>> widths = jnp.ones((3, 4))  # 4 bins
        >>> heights = jnp.ones((3, 4))
        >>> slopes = jnp.ones((3, 3))  # 3 internal knots
        >>> y, log_det = rational_quadratic_spline(x, widths, heights, slopes)
    """
    num_bins = bin_widths.shape[-1]
    dtype = jnp.result_type(inputs)
    # Adaptive tolerance based on machine epsilon, scaled during use
    eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    tiny = jnp.asarray(10.0, dtype=dtype) * eps

    # Normalize widths and heights using softmax
    bin_widths = nnx.softmax(bin_widths, axis=-1)
    bin_heights = nnx.softmax(bin_heights, axis=-1)

    # Enforce minimum bin size
    bin_widths = min_bin_width + (1 - min_bin_width * num_bins) * bin_widths
    bin_heights = min_bin_height + (1 - min_bin_height * num_bins) * bin_heights

    # Compute knot positions (cumulative sum gives knot positions)
    knot_x = jnp.pad(
        jnp.cumsum(bin_widths, -1),
        [(0, 0)] * (bin_widths.ndim - 1) + [(1, 0)],
        constant_values=0,
    )
    knot_y = jnp.pad(
        jnp.cumsum(bin_heights, -1),
        [(0, 0)] * (bin_heights.ndim - 1) + [(1, 0)],
        constant_values=0,
    )

    # Ensure positive slopes using softplus for internal knots
    softplus_scale = np.log(2) / (1 - min_slope)
    internal_slopes = (
        min_slope + nnx.softplus(softplus_scale * knot_slopes) / softplus_scale
    )

    # Pad with 1s for boundary slopes to match linear tails
    padding = [(0, 0)] * (internal_slopes.ndim - 1) + [(1, 1)]
    slopes = jnp.pad(internal_slopes, padding, constant_values=1.0)

    # Handle inputs outside the [0, 1] spline domain,
    # where the transform is the identity.
    in_bounds = (inputs >= 0) & (inputs <= 1)
    # Clamp inputs for internal calculations to avoid NaNs.
    inputs_clipped = jnp.clip(inputs, 0, 1)

    # Helper function for advanced indexing over the batch dimension

    # Find which bin each input falls into, using the clipped inputs.
    if inverse:
        bin_idx = jnp.searchsorted(knot_y, inputs_clipped, side="right") - 1
    else:
        bin_idx = jnp.searchsorted(knot_x, inputs_clipped, side="right") - 1
    # Guard against rightmost insertion at exactly 1.0
    bin_idx = jnp.clip(bin_idx, 0, num_bins - 1)

    # Get bin boundaries and slopes for each input
    left_knot_x = knot_x[bin_idx]
    bin_width = bin_widths[bin_idx]
    left_knot_y = knot_y[bin_idx]
    left_slope = slopes[bin_idx]
    bin_height = bin_heights[bin_idx]
    right_slope = slopes[bin_idx + 1]

    # Compute bin slope (average rise over run)
    # Use a safety guard on width
    width_safe = jnp.maximum(bin_width, tiny)
    bin_slope = bin_height / width_safe

    if inverse:
        # Solve quadratic equation for inverse transform
        y_offset = inputs_clipped - left_knot_y

        # fmt: off
        quad_a = (
            y_offset * (left_slope + right_slope - 2 * bin_slope)
            + bin_height * (bin_slope - left_slope)
        )
        # fmt: off
        quad_b = (
            bin_height * left_slope
            - y_offset * (left_slope + right_slope - 2 * bin_slope)
        )
        quad_c = -bin_slope * y_offset

        discriminant = quad_b**2 - 4 * quad_a * quad_c
        # Discriminant can become slightly negative from roundoff; clamp at 0
        discriminant = jnp.maximum(discriminant, jnp.asarray(0.0, dtype))
        denom_safe = -quad_b - jnp.sqrt(discriminant)
        # Stabilize division by adding a relative tiny term
        denom_safe = denom_safe + tiny * (1.0 + jnp.abs(quad_b))
        normalized_pos = (2 * quad_c) / denom_safe
        normalized_pos = jnp.clip(normalized_pos, 0.0, 1.0)
        outputs_spline = normalized_pos * bin_width + left_knot_x

        # Compute log determinant for the forward transform dy/dx
        pos_complement = normalized_pos * (1 - normalized_pos)
        denominator = bin_slope + (
            (left_slope + right_slope - 2 * bin_slope) * pos_complement
        )
        # Scale lower bound by local magnitudes to preserve units
        denom_lb = tiny * (
            1.0 + jnp.abs(bin_slope) + jnp.abs(left_slope) + jnp.abs(right_slope)
        )
        denominator = jnp.maximum(denominator, denom_lb)
        numerator = bin_slope**2 * (
            right_slope * normalized_pos**2
            + 2 * bin_slope * pos_complement
            + left_slope * (1 - normalized_pos) ** 2
        )
        log_det_spline = jnp.log(numerator) - 2 * jnp.log(denominator)

        # For the inverse transform, we need -log_det(dy/dx)
        log_det = -log_det_spline

    else:
        # Forward transform
        normalized_pos = (inputs_clipped - left_knot_x) / (width_safe)
        normalized_pos = jnp.clip(normalized_pos, 0.0, 1.0)

        # fmt: off
        numerator_term = (
            bin_slope * normalized_pos**2
            + left_slope * normalized_pos * (1 - normalized_pos)
        )
        # fmt: off
        denominator_term = (
            bin_slope
            + (right_slope + left_slope - 2 * bin_slope)
            * normalized_pos * (1 - normalized_pos)
        )
        denom_lb = tiny * (
            1.0 + jnp.abs(bin_slope) + jnp.abs(left_slope) + jnp.abs(right_slope)
        )
        denominator_term = jnp.maximum(denominator_term, denom_lb)
        outputs_spline = left_knot_y + bin_height * numerator_term / denominator_term

        derivative_numerator = bin_slope**2 * (
            right_slope * normalized_pos**2
            + 2 * bin_slope * normalized_pos * (1 - normalized_pos)
            + left_slope * (1 - normalized_pos) ** 2
        )
        log_det = jnp.log(derivative_numerator) - 2 * jnp.log(denominator_term)

    # For out-of-bounds inputs, the transform is the identity.
    # The output is the input, and the log_det is 0.
    outputs = jnp.where(in_bounds, outputs_spline, inputs)
    final_log_det = jnp.where(in_bounds, log_det, jnp.asarray(0.0, dtype=outputs.dtype))

    return outputs, final_log_det


class MonotoneRQSpline(ApplyBijection):
    r"""Monotonic rational quadratic spline bijection.

    Implements element-wise rational quadratic spline transformations that
    maintain monotonicity through parameter normalization. Each element
    is transformed independently using the same spline but different parameters.

    Type: [0, 1]^n → [0, 1]^n (with linear extension outside)

    The spline divides the [0,1] interval into bins and constructs smooth
    rational quadratic curves between knot points. Parameters are automatically
    normalized to ensure monotonicity and numerical stability.

    Args:
        knots: Number of knot points (creates knots-1 bins).
        event_shape: Shape of individual events being transformed.
        min_bin_width: Minimum bin width for numerical stability.
        min_bin_height: Minimum bin height for numerical stability.
        min_slope: Minimum internal knot slope for numerical stability.
        widths_init: Initializer for bin width parameters.
        heights_init: Initializer for bin height parameters.
        slopes_init: Initializer for internal slope parameters.
        rngs: Random number generators for parameter initialization.

    Example:
        >>> spline = MonotoneRQSpline(
        ...     knots=8, event_shape=(3,), rngs=rngs
        ... )
        >>> x = jnp.array([[0.2, 0.5, 0.8]])
        >>> y, log_det = spline.forward(x, jnp.zeros(1))
    """

    def __init__(
        self,
        knots,
        event_shape=(),
        *,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_slope=1e-3,
        widths_init=nnx.initializers.normal(),
        heights_init=nnx.initializers.normal(),
        slopes_init=nnx.initializers.normal(),
        rngs: nnx.Rngs,
    ):
        """Initialize monotonic rational quadratic spline.

        Creates trainable parameters for bin widths, heights, and internal
        knot slopes. Boundary slopes are fixed to 1.0 for linear tails.
        """
        self.event_shape = event_shape
        self.in_features = np.prod(event_shape, dtype=int)

        widths = widths_init(rngs.params(), (*event_shape, knots))
        heights = heights_init(rngs.params(), (*event_shape, knots))
        slopes = slopes_init(rngs.params(), (*event_shape, knots - 1))

        self.widths = nnx.Param(widths)
        self.heights = nnx.Param(heights)
        self.slopes = nnx.Param(slopes)

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_slope = min_slope
        self.knots = knots

    @property
    def param_count(self):
        """Total number of parameters: widths + heights + internal slopes."""
        return 3 * self.knots - 1

    @property
    def param_splits(self):
        """Parameter split sizes for widths, heights, and internal slopes.

        Returns:
            List of sizes [knots, knots, knots-1] for splitting a flattened
            parameter vector into width, height, and slope components.
        """
        return [self.knots, self.knots, self.knots - 1]

    def apply(self, x, log_density, reverse, **kwargs):
        """Apply rational quadratic spline transformation.

        Transforms input through the spline bijection and updates log density
        with the log absolute Jacobian determinant.

        Args:
            x: Input array to transform.
            log_density: Current log density values.
            reverse: Whether to apply inverse transformation.
            **kwargs: Additional arguments (unused).

        Returns:
            Tuple of (transformed_x, updated_log_density).
        """
        event_dim = jnp.ndim(x) - jnp.ndim(log_density)
        si = ShapeInfo(event_dim=event_dim, channel_dim=0)
        _, si = si.process_event(jnp.shape(x))

        y, log_jac = rational_quadratic_spline(
            x,
            self.widths.value,
            self.heights.value,
            self.slopes.value,
            inverse=reverse,
        )

        return y, log_density - jnp.sum(log_jac, axis=si.event_axes)
