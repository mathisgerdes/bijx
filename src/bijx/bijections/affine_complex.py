r"""Complex affine bijections.

A single arithmetic kernel and a thin diagonal-parameter wrapper used by
both :class:`ComplexScaling` and the Fourier-space scalings.

Transform: $y = s \cdot e^{i\varphi} \cdot x + b$.

The log-Jacobian contribution is always $\sum_i w_i \log|s_i|$ for some
caller-supplied weighting $w$ (1 per real DoF, 2 per complex DoF, sums of
rFFT multiplicities for spectrum scaling, etc.). The kernel takes the
precomputed scalar to keep it free of branches.
"""

import jax.numpy as jnp
from flax import nnx

from .base import Bijection

__all__ = [
    "complex_affine_apply",
    "ComplexScaling",
]


def complex_affine_apply(
    x,
    log_density,
    *,
    scale=None,
    phase=None,
    shift=None,
    delta_ld=None,
    invert=False,
):
    r"""Stateless complex affine transform.

    Forward: $y = s \cdot e^{i\varphi} \cdot x + b$.
    Reverse: $x = (y - b) \cdot e^{-i\varphi} / s$.

    Args:
        x: Input array (real or complex).
        log_density: Log-density accumulator.
        scale: Multiplicative factor, broadcastable to ``x``. Applied as-is
            (no abs/exp); may be real or complex.
        phase: Angle in radians, broadcastable to ``x``. Applied as
            multiplication by ``exp(1j * phase)``.
        shift: Additive term, broadcastable to ``x``.
        delta_ld: Precomputed forward log-Jacobian contribution
            $\sum_i w_i \log|s_i|$. The accumulator is updated by
            $-\Delta$ in the forward direction and $+\Delta$ in the
            reverse direction.
        invert: If True, applies the inverse transform.

    Returns:
        Tuple ``(y, log_density)``.
    """
    if not invert:
        if scale is not None:
            x = x * scale
        if phase is not None:
            x = x * jnp.exp(1j * phase)
        if shift is not None:
            x = x + shift
        if delta_ld is not None:
            log_density = log_density - delta_ld
    else:
        if shift is not None:
            x = x - shift
        if phase is not None:
            x = x * jnp.exp(-1j * phase)
        if scale is not None:
            x = x / scale
        if delta_ld is not None:
            log_density = log_density + delta_ld
    return x, log_density


class ComplexScaling(Bijection):
    r"""Diagonal complex affine bijection.

    Transform: $y = e^{s} \cdot e^{i\varphi} \cdot x + b$, with optional
    shift $b$, optional phase $\varphi$, and an optional ``complex_mask``
    selecting which entries carry an imaginary degree of freedom. The
    log-Jacobian contribution is $\sum_i w_i s_i$ with $w_i = 2$ in
    fully-complex layout or $w_i = 1 + \text{mask}_i$ when a mask is
    given.

    For shift and phase, ``..._init=None`` (default) disables the term.
    Pass an initializer (e.g. ``nnx.initializers.zeros``) to enable it,
    or pass a pre-built :class:`nnx.Variable` / :class:`GroupedParam` via
    the corresponding ``shift=`` / ``phase=`` / ``scale=`` argument to
    override construction entirely.

    Index sharing across the event (parameter tying within label groups)
    is supported by passing a :class:`GroupedParam` for any of the terms;
    see :meth:`GroupedParam.from_int_index`. The bijection itself does
    not know about grouping — anything with a ``.get_value()`` returning
    an array broadcastable to ``shape`` works.

    Args:
        shape: Event shape; the layout the bijection acts on.
        scale: Optional pre-built parameter overriding ``scale_init``.
        shift: Optional pre-built parameter; if ``None`` and
            ``shift_init is None``, no shift term is added.
        phase: Optional pre-built parameter; if ``None`` and
            ``phase_init is None``, no phase term is added.
        scale_init: Initializer for the unconstrained log-scale.
        shift_init: Initializer returning shape ``(2, *shape)`` (real/imag
            stacked); ``None`` disables the shift.
        phase_init: Initializer for the phase angle; ``None`` disables
            the phase.
        complex_mask: Optional 0/1 array broadcastable to ``shape`` marking
            which entries carry an imaginary DoF. When provided, phase and
            imaginary shift component are masked on real entries and the
            log-Jacobian weight becomes ``1 + mask``.
        rngs: nnx random number generators (required when any term needs
            initialization).
    """

    def __init__(
        self,
        shape,
        *,
        scale: nnx.Variable | None = None,
        shift: nnx.Variable | None = None,
        phase: nnx.Variable | None = None,
        scale_init=nnx.initializers.normal(),
        shift_init=None,
        phase_init=None,
        complex_mask=None,
        rngs=None,
    ):
        shape = tuple(shape)
        self.shape = shape
        self.complex_mask = complex_mask

        if scale is None:
            scale = nnx.Param(scale_init(rngs.params(), shape))
        self.scale = scale

        if shift is None and shift_init is not None:
            shift = nnx.Param(shift_init(rngs.params(), (2, *shape)))
        self.shift = shift

        if phase is None and phase_init is not None:
            phase = nnx.Param(phase_init(rngs.params(), shape))
        self.phase = phase

    def _resolve_mask(self, kwargs):
        cm = kwargs.pop("complex_mask", None)
        return cm if cm is not None else self.complex_mask

    def _params(self, cm):
        log_s = self.scale.get_value()
        scale = jnp.exp(log_s)
        weight = 2.0 if cm is None else (1 + cm)
        delta_ld = jnp.sum(weight * log_s)

        if self.phase is not None:
            phase = self.phase.get_value()
            if cm is not None:
                phase = phase * cm
        else:
            phase = None

        if self.shift is not None:
            rs = self.shift.get_value()
            real, imag = rs[0], rs[1]
            if cm is not None:
                imag = imag * cm
            shift = real + 1j * imag
        else:
            shift = None

        return scale, phase, shift, delta_ld

    def forward(self, x, log_density, **kwargs):
        cm = self._resolve_mask(kwargs)
        scale, phase, shift, delta_ld = self._params(cm)
        return complex_affine_apply(
            x,
            log_density,
            scale=scale,
            phase=phase,
            shift=shift,
            delta_ld=delta_ld,
            invert=False,
        )

    def reverse(self, x, log_density, **kwargs):
        cm = self._resolve_mask(kwargs)
        scale, phase, shift, delta_ld = self._params(cm)
        return complex_affine_apply(
            x,
            log_density,
            scale=scale,
            phase=phase,
            shift=shift,
            delta_ld=delta_ld,
            invert=True,
        )
