r"""Fundamental bijection abstractions and composition utilities.

This module defines the core bijection interface and provides essential building
blocks for constructing normalizing flows. All bijections implement forward/reverse
transformations with log-density tracking.

Key concepts:
- Bijection: Base class for all invertible transformations
- Chain: Sequential composition of multiple bijections
- Inverse: Reverses the direction of any bijection
- Identity: No-op transformation (useful for composition)

Mathematical background:
Bijections transform both data $\mathbf{x}$ and log-densities $\log p(\mathbf{x})$
according to the change of variables formula:
$$\log p(\mathbf{y}) = \log p(\mathbf{x}) - \log |\det \mathbf{J}|$$
where $\mathbf{J}$ is the Jacobian matrix of the transformation.
"""

from functools import partial

import jax
from flax import nnx

from ..utils import Const


class Bijection(nnx.Module):
    """Base class for all bijective transformations.

    A bijection represents an invertible mapping between two spaces, equipped
    with forward and reverse transformations that properly track log-density
    changes according to the change of variables formula.

    Note:
        This is an abstract base class. Subclasses must override forward() and
        reverse() methods to implement the actual transformation logic.
    """

    def forward(self, x, log_density, **kwargs):
        """Apply forward transformation.

        Transforms input through the bijection and updates log-density according
        to the change of variables formula.

        For convenience ``Bijection()`` gives the default identity bijection.

        Args:
            x: Input data of any pytree structure.
            log_density: Log density values corresponding to the input.
            **kwargs: Additional transformation-specific arguments.

        Returns:
            Tuple of (transformed_data, updated_log_density) where the log-density
            incorporates the log absolute determinant of the transformation Jacobian.
        """
        return x, log_density

    def reverse(self, x, log_density, **kwargs):
        """Apply reverse (inverse) transformation.

        Transforms input through the inverse bijection and updates log-density
        accordingly.

        Args:
            x: Input data of any pytree structure.
            log_density: Log density values corresponding to the input.
            **kwargs: Additional transformation-specific arguments.

        Returns:
            Tuple of (inverse_transformed_data, updated_log_density) where the
            log-density change has the opposite sign compared to forward().
        """
        return x, log_density

    def invert(self):
        """Create an inverted version of this bijection.

        Returns:
            New bijection where forward() and reverse() methods are swapped.
            See :class:`Inverse`.
        """
        return Inverse(self)

    def __call__(self, x, log_density, **kwargs):
        """Apply bijection using forward transformation by default."""
        return self.forward(x, log_density, **kwargs)


Identity = Bijection()
"""Pre-instantiated identity bijection for convenience.

This bijection performs no transformation and can be used as a placeholder
or neutral element in bijection compositions.
"""


class ApplyBijection(Bijection):
    """Convenience base class for bijections with unified forward/reverse logic.

    This class is useful when forward and reverse transformations share most
    of their implementation.
    Instead of duplicating code in separate forward() and reverse() methods,
    subclasses implement a single apply() method with a ``reverse`` parameter.

    Example:
        >>> class MyBijection(ApplyBijection):
        ...     def apply(self, x, log_density, reverse=False, **kwargs):
        ...         # Shared transformation logic here; simple example
        ...         transformed_x = x - 1  if reverse else x + 1
        ...         # log_density doesn't change for this example
        ...         return transformed_x, log_density
    """

    def apply(self, x, log_density, reverse=False, **kwargs):
        """Unified transformation method.

        Args:
            x: Input data of any pytree structure.
            log_density: Log density values corresponding to the input.
            reverse: If True, apply reverse transformation; if False, forward.
            **kwargs: Additional transformation-specific arguments.

        Returns:
            Tuple of (transformed_data, updated_log_density).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def forward(self, x, log_density, **kwargs):
        return self.apply(x, log_density, reverse=False, **kwargs)

    def reverse(self, x, log_density, **kwargs):
        return self.apply(x, log_density, reverse=True, **kwargs)

    def invert(self):
        return Inverse(self)


class CondInverse(Bijection):
    """Conditionally inverted bijection based on runtime boolean flag.

    This bijection wraps another bijection and conditionally inverts it
    based on a boolean parameter. Importantly, the boolean value does not
    have to be known at compile time. Thus, forward/reverse must not change
    array shapes.

    Args:
        bijection: The underlying bijection to wrap.
        invert: If True, swap forward/reverse directions.
    """

    def __init__(self, bijection: Bijection, invert: bool = True):
        self.invert = Const(invert)
        self.bijection = bijection

    def forward(self, x, log_density, **kwargs):
        return jax.lax.cond(
            self.invert.value,
            lambda x, ld, kw: self.bijection.reverse(x, ld, **kw),
            lambda x, ld, kw: self.bijection.forward(x, ld, **kw),
            x,
            log_density,
            kwargs,
        )

    def reverse(self, x, log_density, **kwargs):
        return jax.lax.cond(
            self.invert.value,
            lambda x, ld, kw: self.bijection.forward(x, ld, **kw),
            lambda x, ld, kw: self.bijection.reverse(x, ld, **kw),
            x,
            log_density,
            kwargs,
        )

    def invert(self):
        return CondInverse(self.bijection, ~self.invert.value)


class Inverse(Bijection):
    """Inverted bijection that swaps forward and reverse directions.

    This bijection wraps another bijection and swaps its forward()
    and reverse() methods.

    Args:
        bijection: The bijection to invert.

    Note:
        Inverses of bijections should usually be created using ``bij.invert()``.
        Then, ``bijx.invert().invert() == bij``.
        Otherwise, ``Inverse(Inverse(bij))`` while acting the same as ``bij`` does not
        "undo" the unnecessary wrapping.
    """

    def __init__(self, bijection: Bijection):
        self.bijection = bijection

    def forward(self, x, log_density, **kwargs):
        return self.bijection.reverse(x, log_density, **kwargs)

    def reverse(self, x, log_density, **kwargs):
        return self.bijection.forward(x, log_density, **kwargs)

    def invert(self):
        """Return the original bijection (double inversion)."""
        return self.bijection


class Chain(Bijection):
    r"""Sequential composition of multiple bijections.

    Chains multiple bijections together to create a composite transformation.
    Forward pass applies bijections in order, reverse pass applies them in
    reverse order with each bijection's reverse() method.

    Args:
        *bijections: Variable number of bijections to chain together.

    Example:
        >>> bij1 = SomeBijection()
        >>> bij2 = SomeBijection()
        >>> chain = Chain(bij1, bij2)
        >>> # Forward: bij2.forward(bij1.forward(x, ld))
        >>> # Reverse: bij1.reverse(bij2.reverse(y, ld))
    """

    def __init__(self, *bijections: Bijection):
        self.bijections = nnx.List(bijections)

    def forward(self, x, log_density, *, arg_list: list[dict] | None = None, **kwargs):
        """Apply all bijections in forward order.

        Args:
            x: Input data.
            log_density: Input log density.
            arg_list: Optional list of argument dicts for each bijection.
            **kwargs: Common arguments passed to all bijections.

        Returns:
            Tuple of final transformed data and accumulated log density.
        """
        if arg_list is None:
            arg_list = [{}] * len(self.bijections)
        for bijection, args in zip(self.bijections, arg_list, strict=True):
            x, log_density = bijection.forward(x, log_density, **args, **kwargs)
        return x, log_density

    def reverse(self, x, log_density, *, arg_list: list[dict] | None = None, **kwargs):
        """Apply all bijections in reverse order using their reverse() methods.

        Args:
            x: Input data.
            log_density: Input log density.
            arg_list: Optional list of argument dicts for each bijection.
            **kwargs: Common arguments passed to all bijections.

        Returns:
            Tuple of final inverse-transformed data and accumulated log density.
        """
        if arg_list is None:
            arg_list = [{}] * len(self.bijections)
        for bijection, args in zip(
            reversed(self.bijections), reversed(arg_list), strict=True
        ):
            x, log_density = bijection.reverse(x, log_density, **args, **kwargs)
        return x, log_density


class ScanChain(Bijection):
    r"""Jax compilation-efficient chain of identical bijections using JAX scan.

    This bijection applies the same bijection architecture multiple times
    in sequence, but with different parameters for each application by using
    ``jax.lax.scan`` over the stack of bijections for efficient jax-compilation.

    Args:
        stack: Stack of bijections to scan over. Should be a single bijection
            but all internal parameters carry an initial "scan batch" dimension.

    Note:
        The stack should contain parameters for multiple instances of the
        same bijection architecture. Forward pass scans in order, reverse
        pass scans in reverse order. The "scan" index is the leading dimension
        of all internal parameters.
    """

    def __init__(self, stack: Bijection):
        self.stack = stack

    def _forward(self, carry, variables, graph, **kwargs):
        bijection = nnx.merge(graph, variables)
        return bijection.forward(*carry, **kwargs), None

    def _reverse(self, carry, variables, graph, **kwargs):
        bijection = nnx.merge(graph, variables)
        return bijection.reverse(*carry, **kwargs), None

    def forward(self, x, log_density, **kwargs):
        graph, variables = nnx.split(self.stack)
        (y, lp), _ = jax.lax.scan(
            partial(self._forward, graph=graph, **kwargs),
            (x, log_density),
            variables,
        )
        return y, lp

    def reverse(self, y, log_density, **kwargs):
        graph, variables = nnx.split(self.stack)
        (x, lp), _ = jax.lax.scan(
            partial(self._reverse, graph=graph, **kwargs),
            (y, log_density),
            variables,
            reverse=True,
        )
        return x, lp


class Frozen(Bijection):
    """Wrapper to screen internal parameters from training.

    To be used in conjunction with :obj:`FrozenFilter`.

    Args:
        bijection: The bijection to freeze.

    Note:
        The use of this wrapper comes purely from using it in conjunction with
        a filter; the internal parameters of the bijection module are not
        modified in any way.
    """

    def __init__(self, bijection: Bijection):
        self.frozen = bijection

    def forward(self, x, log_density, **kwargs):
        return self.frozen.forward(x, log_density, **kwargs)

    def reverse(self, x, log_density, **kwargs):
        return self.frozen.reverse(x, log_density, **kwargs)
