r"""Radial bijections for normalizing flows.

This module provides radial transformations that operate on multi-dimensional
spaces by transforming the radial coordinate while preserving angular structure.
These bijections are particularly useful for modeling distributions with radial
symmetry or implementing flows with spherical structure.

Key components:
    - :class:`RayTransform`: Ensures $f(0) = 0$ property for any bijection
    - :class:`Radial`: Basic radial flow with learnable scaling and centering
    - :class:`RadialConditional`: Radial flow with angle-dependent parameters

The radial transformations follow the pattern $g(x) = c + f(\abs{S(x-c)}) v$,
where v is the unit vector in the direction of $S(x-c)$, $S$ is a diagonal scaling
matrix, c is a center vector, and f is a scalar bijection that transforms the
radius.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from ..utils import ParamSpec, default_wrap
from .base import ApplyBijection, Bijection
from .coupling import ModuleReconstructor


class RayTransform(Bijection):
    """Simple ray transformation ensuring f(0)=0."""

    def __init__(self, bijection: Bijection):
        self.bijection = bijection

    def forward(self, x, ld):
        offset, _ = self.bijection.forward(jnp.zeros(()), jnp.zeros(()))
        y, ld = self.bijection.forward(x, ld)
        return y - offset, ld

    def reverse(self, y, ld):
        offset, _ = self.bijection.forward(jnp.zeros(()), jnp.zeros(()))
        return self.bijection.reverse(y + offset, ld)


class Radial(ApplyBijection):
    r"""Radial bijection with learnable scaling and centering.

    This bijection implements a radial transformation $g(x) = c + f(\abs{S(x-c)}) * v$,
    where v is the unit vector $(S(x-c))/\abs{S(x-c)}$, S is a diagonal scaling
    matrix, and c is a center vector. The scalar function $f: r \rightarrow r'$ is
    provided by another bijection.

    The transformation is composed as:
    1. Centering and scaling: $y = S(x-c)$
    2. Radial transformation on $y$.
    3. Un-scaling and un-centering.

    The log-determinant contributions from the scaling $S$ cancel out, so only
    the radial transformation's Jacobian determinant is included.

    Args:
        scalar_bijection: A bijection that transforms a scalar radius.
        n_dims: The dimensionality of the space.
        center: Initial center vector $c$. If None, defaults to zeros.
        scale: Initial scale vector for $S$. If None, defaults to ones.
    """

    def __init__(
        self,
        scalar_bijection: Bijection,
        center: ParamSpec = (),
        scale: ParamSpec = (),
        rngs: nnx.Rngs = None,
    ):
        self.scalar_bijection = scalar_bijection

        self.center = default_wrap(
            center, init_fn=nnx.initializers.normal(1), rngs=rngs
        )
        self.log_scale = default_wrap(
            scale, init_fn=nnx.initializers.normal(1), rngs=rngs
        )

    @property
    def scale(self) -> jax.Array:
        """Positive scaling factors."""
        return jnp.exp(self.log_scale.value)

    def apply(
        self, x: jax.Array, log_density: jax.Array, reverse: bool = False, **kwargs
    ):
        """Apply radial transformation, forward or reverse."""

        x_centered = x - self.center.value
        x_scaled = x_centered * self.scale

        r_in = jnp.linalg.norm(x_scaled, axis=-1, keepdims=False)
        r_safe = jnp.where(r_in > 0, r_in, 1)

        if reverse:
            r_out, mld_scalar = self.scalar_bijection.reverse(
                r_in, jnp.zeros_like(r_in)
            )
            ratio = jnp.abs(r_out / r_safe)
        else:  # forward
            r_out, mld_scalar = self.scalar_bijection.forward(
                r_in, jnp.zeros_like(r_in)
            )
            ratio = jnp.abs(r_out / r_safe)

        # Stable evaluation of log(f(r)/r) with correct r→0 limit log f'(0)
        log_ratio_term = jnp.where(
            r_in > 0,
            # mld_scalar + jnp.log1p(jnp.exp(-mld_scalar) * ratio - 1.0),
            jnp.log(ratio),
            -mld_scalar,
        )

        y_scaled = jnp.expand_dims(ratio, -1) * x_scaled
        y_centered = y_scaled / self.scale
        y = y_centered + self.center.value

        # Log-determinant of the Jacobian: log f'(r) + (n-1) log(f(r)/r)
        log_det_radial = mld_scalar + (1 - x.shape[-1]) * log_ratio_term

        return y, log_density + log_det_radial


class RadialConditional(ApplyBijection):
    r"""Radial flow with angle-dependent bijection parameters.

    Extends basic radial flows by letting $r -> f(r; theta(x/\abs{x}))$.
    The conditioning network maps unit vectors to bijection parameters.

    Args:
        scalar_bijection: Base bijection for radial transformation.
        cond_net: Conditioning network.
        center: Center of the radial transformation.
        scale: Scale factors for each dimension.
        rngs: Random number generators.
    """

    def __init__(
        self,
        scalar_bijection: Bijection,
        cond_net: nnx.Module,
        center: ParamSpec = (),
        scale: ParamSpec = (),
        rngs: nnx.Rngs = None,
    ):
        self.center = default_wrap(
            center, init_fn=nnx.initializers.normal(1), rngs=rngs
        )
        self.log_scale = default_wrap(
            scale, init_fn=nnx.initializers.normal(1), rngs=rngs
        )

        self.reconst = ModuleReconstructor(scalar_bijection)
        self.cond_net = cond_net

    @property
    def scale(self) -> jax.Array:
        """Positive scaling factors."""
        return jnp.exp(self.log_scale.value)

    def apply(
        self, x: jax.Array, log_density: jax.Array, reverse: bool = False, **kwargs
    ):
        """Apply radial transformation, forward or reverse."""

        x_centered = x - self.center.value
        x_scaled = x_centered * self.scale

        r_in = jnp.linalg.norm(x_scaled, axis=-1, keepdims=False)
        r_safe = jnp.where(r_in > 0, r_in, 1)

        x_hat = x_scaled / jnp.expand_dims(r_safe, -1)
        params = self.cond_net(x_hat)
        bijection = self.reconst.from_params(params, autovmap=True)

        if reverse:
            r_out, mld_scalar = bijection.reverse(r_in, jnp.zeros_like(r_in))
            ratio = jnp.abs(r_out / r_safe)
        else:  # forward
            r_out, mld_scalar = bijection.forward(r_in, jnp.zeros_like(r_in))
            ratio = jnp.abs(r_out / r_safe)

        # Stable evaluation of log(f(r)/r) with correct r→0 limit log f'(0)
        log_ratio_term = jnp.where(
            r_in > 0,
            jnp.log(ratio),
            -mld_scalar,
        )

        y_scaled = jnp.expand_dims(ratio, -1) * x_scaled
        y_centered = y_scaled / self.scale
        y = y_centered + self.center.value

        # Log-determinant of the Jacobian: log f'(r) + (n-1) log(f(r)/r)
        log_det_radial = mld_scalar + (1 - x.shape[-1]) * log_ratio_term

        return y, log_density + log_det_radial
