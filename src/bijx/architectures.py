r"""Ready-to-use example flow architectures (coupling-based).

This module packages the recurring ~60-line composition of
:class:`~bijx.GeneralCouplingLayer`, :class:`~bijx.ModuleReconstructor`,
:func:`~bijx.stack_bijections` and :func:`~bijx.extract_init` into one-call
builders, with the *defaults* validated by the scalar-parametrization study
(see ``sharp-bits.md`` and the study's ``report.md``):

- :func:`coupling_flow` -- NSF / spline / analytic-bijection style coupling flow:
  alternating checkerboard masks, a per-element scalar transform built as a
  :class:`~bijx.ScanChain` of ``n_copies`` copies of a user bijection, and a
  ResNet conditioner whose final-layer bias can carry a near-identity init via
  :func:`~bijx.extract_init`.
- :func:`realnvp_flow` -- the classic RealNVP affine-coupling flow (the same
  plumbing with :class:`~bijx.AffineLinear` as the per-element transform).
- :func:`realnvp_conv_flow` -- a convolutional RealNVP variant for image-like
  (``H, W, C``) events, using a :class:`~bijx.nn.nets.ConvNet` conditioner.
- :func:`init_for` -- a regime/init helper returning a bijection *factory* with
  the study-validated role-separated, depth-aware init baked in (so users do not
  re-derive the presets).

The builders return a plain :class:`~bijx.Chain` of coupling layers; wrap it in
:class:`~bijx.Transformed` over a base distribution to obtain a trainable flow::

    flow = bijx.coupling_flow(2, lambda rngs: bijx.MonotoneRQSpline(8, (), rngs=rngs),
                              rngs=nnx.Rngs(0))
    model = bijx.Transformed(bijx.IndependentNormal((2,)), flow)
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx

from .bijections.analytic import (
    CubicConjugation,
    CubicRational,
    SinhConjugation,
    _safe_exp_scale_inv,
    _softplus_inv,
    safe_exp_scale,
)
from .bijections.base import Chain, ScanChain
from .bijections.coupling import (
    GeneralCouplingLayer,
    ModuleReconstructor,
    checker_mask,
    extract_init,
    stack_bijections,
)
from .nn.nets import ConvNet, ResNet

__all__ = [
    "coupling_flow",
    "realnvp_flow",
    "realnvp_conv_flow",
    "init_for",
]


def _bijection_stack(bijection_factory: Callable, n_copies: int):
    """Return a generator ``rngs -> Bijection`` for a ScanChain of ``n_copies``.

    With ``n_copies < 2`` this is a single (non-stacked) bijection; with
    ``n_copies >= 2`` it is a :class:`~bijx.ScanChain` of independently
    initialized copies (see :func:`~bijx.stack_bijections`).

    Note: ``ScanChain`` is only applied for ``n_copies >= 2`` -- scanning a
    single (non-stacked) module whose parameter leaves have heterogeneous
    leading sizes fails (see ``sharp-bits.md``).
    """
    transform = ScanChain if n_copies >= 2 else (lambda b: b)
    return stack_bijections(
        lambda rngs: bijection_factory(rngs=rngs),
        transform=transform,
        copies=n_copies,
    )


def _tiled_extract_bias_init(bijection_factory, n_copies, count_active):
    """Final-layer ``bias_init`` = the bijection's own initialized raw params,
    tiled across the active elements.

    With a (near-)zero conditioner kernel, the network output at init equals this
    bias, so each active element starts as a fully spec-initialized bijection
    (the :func:`~bijx.extract_init` mechanism) rather than the inert all-zeros
    params produced by ``bias_mode="zeros"``.
    """
    base = extract_init(_bijection_stack(bijection_factory, n_copies))

    def bias_init(rng, shape, dtype=jnp.float32):
        vec = base(rng)
        return jnp.tile(vec, count_active).astype(dtype)

    return bias_init


def _coupling_layer(
    mask,
    bijection_factory,
    *,
    n_copies,
    width,
    depth,
    conditioner_activation,
    bias_mode,
    final_kernel_init,
    rngs,
):
    count_active, count_passive = mask.counts
    template = ModuleReconstructor(_bijection_stack(bijection_factory, n_copies)(rngs))
    param_count = template.params_total_size

    if bias_mode == "extract_init":
        final_bias_init = _tiled_extract_bias_init(
            bijection_factory, n_copies, count_active
        )
    elif bias_mode == "zeros":
        final_bias_init = nnx.initializers.zeros
    else:
        raise ValueError(f"unknown bias_mode {bias_mode!r}")

    resnet = ResNet(
        count_passive,
        count_active * param_count,
        width,
        depth,
        activation=conditioner_activation,
        final_kernel_init=final_kernel_init,
        final_bias_init=final_bias_init,
        rngs=rngs,
    )

    def reshape_params(p):
        return p.reshape(p.shape[:-1] + (count_active, param_count))

    param_net = nnx.Sequential(resnet, reshape_params)
    return GeneralCouplingLayer(
        param_net, mask, template, bijection_event_rank=0, split=True
    )


def coupling_flow(
    event_size: int,
    bijection_factory: Callable,
    *,
    n_coupling_layers: int = 8,
    n_copies: int = 1,
    width: int = 128,
    depth: int = 2,
    conditioner_activation: Callable = nnx.gelu,
    bias_mode: str = "extract_init",
    final_kernel_init: Callable | None = None,
    parity: bool = True,
    rngs: nnx.Rngs,
) -> Chain:
    r"""Checkerboard coupling flow with a per-element scalar transform.

    Each coupling layer splits a length-``event_size`` vector by an alternating
    checkerboard mask and transforms the active half with a stack of ``n_copies``
    copies of ``bijection_factory`` (a :class:`~bijx.ScanChain` when
    ``n_copies >= 2``). A :class:`~bijx.nn.nets.ResNet` conditioner maps the
    passive half to the per-element bijection parameters.

    Args:
        event_size: Length of the (flat) event vector.
        bijection_factory: Callable ``rngs -> Bijection`` building one scalar
            bijection (e.g. ``lambda rngs: bijx.MonotoneRQSpline(8, (), rngs=rngs)``
            or a factory from :func:`init_for`).
        n_coupling_layers: Number of coupling layers (masks alternate each layer).
        n_copies: Copies of ``bijection_factory`` composed per element.
        width, depth: ResNet conditioner width and number of residual blocks.
        conditioner_activation: ResNet hidden activation.
        bias_mode: ``"extract_init"`` (default) seeds each element with the
            bijection's own init via :func:`~bijx.extract_init` (near-identity if
            the factory's delta-control init is small); ``"zeros"`` uses a zero
            final bias, so identity-at-init must come from the transform's value
            at zero and the factory's own ``*_init`` is inert.
        final_kernel_init: Conditioner final-layer kernel init. Defaults to a
            small ``normal(1e-3)`` under ``extract_init`` (so the bias dominates
            at init) and ``normal(0.01)`` under ``zeros``.
        parity: Starting checkerboard parity.
        rngs: NNX rngs.

    Returns:
        A :class:`~bijx.Chain` of coupling layers. Wrap in
        :class:`~bijx.Transformed` for a full flow.

    Example:
        >>> import bijx, jax.numpy as jnp
        >>> from flax import nnx
        >>> rngs = nnx.Rngs(0)
        >>> flow = bijx.coupling_flow(
        ...     4, lambda rngs: bijx.MonotoneRQSpline(8, (), rngs=rngs),
        ...     n_coupling_layers=4, bias_mode="zeros", rngs=rngs)
        >>> x = jnp.ones((2, 4))
        >>> y, log_det = flow.forward(x, jnp.zeros(2))
        >>> xb, ld = flow.reverse(y, log_det)
        >>> bool(jnp.allclose(xb, x, atol=1e-4))
        True
    """
    if final_kernel_init is None:
        scale = 1e-3 if bias_mode == "extract_init" else 0.01
        final_kernel_init = nnx.initializers.normal(scale)

    mask = checker_mask((event_size,), parity)
    layers = []
    for _ in range(n_coupling_layers):
        layers.append(
            _coupling_layer(
                mask,
                bijection_factory,
                n_copies=n_copies,
                width=width,
                depth=depth,
                conditioner_activation=conditioner_activation,
                bias_mode=bias_mode,
                final_kernel_init=final_kernel_init,
                rngs=rngs,
            )
        )
        mask = ~mask
    return Chain(*layers)


def realnvp_flow(
    event_size: int,
    *,
    n_coupling_layers: int = 8,
    width: int = 128,
    depth: int = 2,
    conditioner_activation: Callable = nnx.gelu,
    bias_mode: str = "zeros",
    final_kernel_init: Callable | None = None,
    parity: bool = True,
    rngs: nnx.Rngs,
) -> Chain:
    r"""RealNVP-style affine coupling flow (vector variant).

    Affine coupling: each active element is transformed by
    :class:`~bijx.AffineLinear` (``a * x + b`` with ``a = exp(scale)``), whose
    parameters come from a ResNet conditioner. ``AffineLinear`` is identity at
    zero parameters, so the default ``bias_mode="zeros"`` already starts the flow
    at identity (no :func:`~bijx.extract_init` needed).

    Args:
        event_size: Length of the (flat) event vector.
        n_coupling_layers, width, depth, conditioner_activation, parity, rngs:
            See :func:`coupling_flow`.
        bias_mode: Defaults to ``"zeros"`` (identity at init for affine coupling).
        final_kernel_init: Conditioner final-layer kernel init. Under the default
            ``zeros`` bias this defaults to ``zeros`` as well, giving an EXACT
            identity at init (the canonical RealNVP start); pass an explicit small
            ``normal(...)`` to perturb away from identity.

    Returns:
        A :class:`~bijx.Chain` of affine coupling layers.

    Example:
        >>> import bijx, jax.numpy as jnp
        >>> from flax import nnx
        >>> flow = bijx.realnvp_flow(4, n_coupling_layers=4, rngs=nnx.Rngs(0))
        >>> x = jnp.linspace(-1, 1, 8).reshape(2, 4)
        >>> y, log_det = flow.forward(x, jnp.zeros(2))
        >>> xb, ld = flow.reverse(y, log_det)
        >>> bool(jnp.allclose(xb, x, atol=1e-5))
        True
        >>> # zeros bias + zeros final kernel => exact identity at init
        >>> bool(jnp.allclose(y, x, atol=1e-6)) and bool(jnp.allclose(log_det, 0.0))
        True
    """
    from .bijections.scalar import AffineLinear

    if final_kernel_init is None and bias_mode == "zeros":
        final_kernel_init = nnx.initializers.zeros

    return coupling_flow(
        event_size,
        lambda rngs: AffineLinear(rngs=rngs),
        n_coupling_layers=n_coupling_layers,
        n_copies=1,
        width=width,
        depth=depth,
        conditioner_activation=conditioner_activation,
        bias_mode=bias_mode,
        final_kernel_init=final_kernel_init,
        parity=parity,
        rngs=rngs,
    )


def realnvp_conv_flow(
    event_shape: tuple[int, ...],
    *,
    n_coupling_layers: int = 8,
    hidden_channels: list[int] | None = None,
    kernel_size: tuple[int, ...] = (3, 3),
    conditioner_activation: Callable = nnx.leaky_relu,
    final_kernel_init: Callable | None = None,
    parity: bool = True,
    rngs: nnx.Rngs,
) -> Chain:
    r"""Convolutional RealNVP affine-coupling flow for image-like events.

    Uses multiplicative (checkerboard) masking over the spatial grid and a
    :class:`~bijx.nn.nets.ConvNet` conditioner. The event is ``(H, W, C)``; the
    conditioner sees the masked input and outputs ``2 * C`` channels (affine
    scale and shift) per pixel. The final ConvNet activation is ``tanh``, and the
    small final-kernel init keeps the scale near zero (identity) at init.

    Args:
        event_shape: ``(H, W, C)`` event shape.
        n_coupling_layers: Number of coupling layers (mask parity alternates).
        hidden_channels: ConvNet hidden channel sizes (default ``[32, 32]``).
        kernel_size: Convolution kernel size.
        conditioner_activation: ConvNet hidden activation.
        final_kernel_init: ConvNet final-layer kernel init (default small normal).
        parity: Starting checkerboard parity.
        rngs: NNX rngs.

    Returns:
        A :class:`~bijx.Chain` of convolutional affine coupling layers.

    Example:
        >>> import bijx, jax.numpy as jnp
        >>> from flax import nnx
        >>> flow = bijx.realnvp_conv_flow((8, 8, 1), n_coupling_layers=2,
        ...                               rngs=nnx.Rngs(0))
        >>> x = jnp.ones((2, 8, 8, 1))
        >>> y, log_det = flow.forward(x, jnp.zeros(2))
        >>> xb, ld = flow.reverse(y, log_det)
        >>> bool(jnp.allclose(xb, x, atol=1e-4))
        True
    """
    from .bijections.scalar import AffineLinear

    if hidden_channels is None:
        hidden_channels = [32, 32]
    if final_kernel_init is None:
        final_kernel_init = nnx.initializers.normal(0.01)

    event_shape = tuple(event_shape)
    *spatial, channels = event_shape
    spatial = tuple(spatial)
    template = ModuleReconstructor(AffineLinear(rngs=rngs))
    param_count = template.params_total_size

    def conv_layer(mask):
        conv = ConvNet(
            in_channels=channels,
            out_channels=channels * param_count,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            activation=conditioner_activation,
            final_kernel_init=final_kernel_init,
            final_bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

        def reshape_params(p):
            # p: (..., H, W, C * param_count) -> (..., H, W, C, param_count)
            return p.reshape(p.shape[:-1] + (channels, param_count))

        param_net = nnx.Sequential(conv, reshape_params)
        return GeneralCouplingLayer(
            param_net, mask, template, bijection_event_rank=0, split=False
        )

    # Multiplicative masking over the FULL (H, W, C) event so the mask broadcasts
    # cleanly against the channel dimension; the conv conditioner sees the masked
    # input and produces per-pixel affine parameters.
    mask = checker_mask(event_shape, parity)
    layers = []
    for _ in range(n_coupling_layers):
        layers.append(conv_layer(mask))
        mask = ~mask
    return Chain(*layers)


# ---------------------------------------------------------------------------
# init_for: regime/init helper (study-validated presets)
# ---------------------------------------------------------------------------

_ANALYTIC_BIJECTIONS = ("cubic", "cubic_rational", "sinh")


def _delta_init(delta0: float, depth: int, delta_scale: str):
    """Raw init for delta-control params: ``normal(delta0 * f(depth))``."""
    if delta_scale == "inv_sqrt":
        f = 1.0 / math.sqrt(depth)
    elif delta_scale == "const":
        f = 1.0
    elif delta_scale == "inv":
        f = 1.0 / depth
    else:
        raise ValueError(
            f"delta_scale must be one of ('inv_sqrt', 'const', 'inv'), "
            f"got {delta_scale!r}"
        )
    return nnx.initializers.normal(delta0 * f)


def _param(init_fn, rngs):
    """Build a scalar ``nnx.Param`` from a Flax initializer."""
    return nnx.Param(init_fn(rngs.params(), ()))


def init_for(
    bijection: str,
    *,
    architecture: str,
    depth: int = 1,
    delta0: float = 0.5,
    delta_scale: str = "inv_sqrt",
):
    r"""Build a study-validated bijection *factory* for the given regime.

    Returns ``rngs -> Bijection`` ready to pass to :func:`coupling_flow` (or to
    :func:`~bijx.stack_bijections` for the stacked regime). The init follows the
    role taxonomy validated by the scalar-parametrization study (see
    ``sharp-bits.md``):

    - **delta-control** params (those that make the map deviate from identity)
      get a small init that shrinks with ``depth`` (``normal(delta0/sqrt(depth))``
      by default) so a deep stack stays near identity at init;
    - **scale** params start at a neutral value (curvature radius / overall
      scale ~ 1, cubic ``a~1, b~0.3``) -- never random, never 0;
    - **location** is 0.

    Regimes:

    - ``architecture="coupling"``: role-separated, depth-aware init. Use with
      ``coupling_flow(..., bias_mode="extract_init")`` so the init actually seeds
      each element. For ``cubic`` the bounded ``safe_exp_scale`` ``a, b`` default
      is used (off-experimental per the study). For ``cubic_rational`` this is the
      recommended preset only at depth >= ~16 (a depth gate, emitted as a
      warning below); at shallow depth one near-singular target was observed.
      For ``sinh`` the conservative variant is used (alpha floor 0.1, zeros
      delta-control), and a note is emitted that sinh coupling is robustness-
      limited at the *family* level.
    - ``architecture="stacked"``: the active (non-timid) finalist init for a
      conditioner-free :class:`~bijx.ScanChain` of scalar bijections. Role/
      near-identity init is intentionally NOT used here (the study found that to
      be an optimization-speed, not expressiveness, issue).

    Args:
        bijection: One of ``"cubic"``, ``"cubic_rational"``, ``"sinh"``.
        architecture: ``"coupling"`` or ``"stacked"``.
        depth: Number of stacked copies / coupling-stack depth (drives the
            delta-control init scaling for the coupling regime).
        delta0: Base scale of the delta-control init.
        delta_scale: ``"inv_sqrt"`` (default), ``"const"`` or ``"inv"`` --
            how the delta-control init shrinks with ``depth``.

    Returns:
        Callable ``rngs -> Bijection``.

    Example:
        >>> import bijx, warnings
        >>> from flax import nnx
        >>> factory = bijx.init_for("cubic", architecture="coupling", depth=8)
        >>> flow = bijx.coupling_flow(4, factory, n_coupling_layers=8, n_copies=2,
        ...                           bias_mode="extract_init", rngs=nnx.Rngs(0))
        >>> import jax.numpy as jnp
        >>> y, ld = flow.forward(jnp.ones((2, 4)), jnp.zeros(2))
        >>> bool(jnp.all(jnp.isfinite(y)))
        True
    """
    import warnings

    if bijection not in _ANALYTIC_BIJECTIONS:
        raise ValueError(
            f"bijection must be one of {_ANALYTIC_BIJECTIONS}, got {bijection!r}"
        )
    if architecture not in ("coupling", "stacked"):
        raise ValueError(
            f"architecture must be 'coupling' or 'stacked', got {architecture!r}"
        )

    if architecture == "stacked":
        # Active finalist init: the per-element defaults (role-correct scale/loc,
        # small-but-not-depth-shrunk delta-control) already match the study's
        # Preset S parametrization for a conditioner-free stack.
        if bijection == "cubic_rational":
            return lambda rngs: CubicRational(rngs=rngs)
        if bijection == "sinh":
            return lambda rngs: SinhConjugation(rngs=rngs)
        return lambda rngs: CubicConjugation(rngs=rngs)

    # architecture == "coupling": role-separated, depth-aware delta init.
    delta = _delta_init(delta0, depth, delta_scale)
    zeros = nnx.initializers.zeros_init()

    if bijection == "cubic_rational":
        if depth < 16:
            warnings.warn(
                "init_for('cubic_rational', architecture='coupling') is the "
                "study-validated preset only at depth >= ~16; at shallow depth a "
                "near-singular target was observed. Consider 'cubic' (off-"
                "experimental at depth 8) or a deeper stack.",
                stacklevel=2,
            )
        beta_raw = float(_softplus_inv(0.9)) - 1.0  # beta ~ 1 through SoftplusTransform

        def factory(rngs):
            return CubicRational(
                alpha=_param(delta, rngs),
                beta=_param(nnx.initializers.constant(beta_raw), rngs),
                loc=_param(zeros, rngs),
                rngs=rngs,
            )

        return factory

    if bijection == "cubic":
        a_raw = float(_safe_exp_scale_inv(1.0))  # a ~ 1 through safe_exp_scale
        b_raw = float(_safe_exp_scale_inv(0.3))  # b ~ 0.3

        def factory(rngs):
            return CubicConjugation(
                beta=_param(delta, rngs),
                a=_param(nnx.initializers.constant(a_raw), rngs),
                b=_param(nnx.initializers.constant(b_raw), rngs),
                loc=_param(zeros, rngs),
                a_transform=safe_exp_scale,
                b_transform=safe_exp_scale,
                rngs=rngs,
            )

        return factory

    # sinh: conservative preset (alpha floor 0.1, zeros delta-control, scale -> 1).
    warnings.warn(
        "init_for('sinh', architecture='coupling') ships the conservative preset. "
        "sinh coupling is usable: with the gradient-correct all-order log-Jac it "
        "trains fine even at the hardest depth (study plan_06b/06c: 0/10 diverged at "
        "depth 16). An earlier 'family-limited' claim was a log-Jac GRADIENT bug, "
        "since fixed -- it is retracted. For very deep stacks keep this conservative "
        "preset (or consider the stacked regime).",
        stacklevel=2,
    )
    alpha_floor = 0.1
    alpha_raw = float(_softplus_inv(1.0 - alpha_floor))

    def factory(rngs):
        return SinhConjugation(
            alpha=_param(nnx.initializers.constant(alpha_raw), rngs),
            loc=_param(zeros, rngs),
            beta=_param(zeros, rngs),
            mu=_param(zeros, rngs),
            nu=_param(zeros, rngs),
            alpha_transform=lambda x: nnx.softplus(x) + alpha_floor,
            rngs=rngs,
        )

    return factory
