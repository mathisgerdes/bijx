"""
FlowJAX compatibility layer for bijx.

This module provides bidirectional compatibility between bijx and FlowJAX,
allowing users to seamlessly integrate bijections and distributions from
both libraries.

Main functions:
    - :func:`from_flowjax`: Convert FlowJAX components to bijx
    - :func:`to_flowjax`: Convert bijx components to FlowJAX

Example:
    >>> import flowjax.bijections as fbij
    >>> flowjax_bij = fbij.RationalQuadraticSpline(knots=5, interval=(-1, 1))
    >>> bijx_bij = from_flowjax(flowjax_bij)
    >>> # Now use bijx_bij with bijx interface
    >>> y, log_det = bijx_bij.forward(x, log_density)


Main differences between the two frameworks include:
    - Different ML libraries (flax.nnx vs equinox)
    - Different shape assumptions (runtime inference vs unbatched assumption)
    - General keywords (bijx) vs specific ``condition`` parameter (flowjax)
"""

import flowjax
import flowjax.bijections
import flowjax.distributions
import jax
import jax.numpy as jnp
from flax import nnx
from jax_autovmap import autovmap

from .bijections.base import Bijection
from .distributions import Distribution


class FlowjaxToBijxBijection(Bijection):
    """Adapter to use FlowJAX bijections with bijx interface.

    Wraps a FlowJAX bijection to implement the bijx :class:`Bijection` interface.

    Note:
        Log determinant signs are flipped between the two libraries:
        bijx subtracts log determinants while FlowJAX adds them.

    Example:
        >>> import flowjax.bijections as fbij
        >>> flowjax_spline = fbij.RationalQuadraticSpline(knots=10, interval=(-1, 1))
        >>> bijx_spline = FlowjaxToBijxBijection(flowjax_spline)
        >>> y, log_det = bijx_spline.forward(x, log_density)
    """

    def __init__(self, flowjax_bijection, cond_name: str = "condition"):
        """Initialize FlowJAX to bijx bijection adapter.

        Args:
            flowjax_bijection: FlowJAX bijection to wrap.
            cond_name: Keyword argument name for passing conditional inputs.
        """
        params, self.treedef = jax.tree.flatten(flowjax_bijection)
        self.params = nnx.Param(params)
        self.cond_name = cond_name

        conditional_shape = flowjax_bijection.cond_shape
        state_shape = flowjax_bijection.shape

        def _apply(bijection, x, condition, log_density, inverse):
            if inverse:
                x, log_det = bijection.inverse_and_log_det(x, condition)
            else:
                x, log_det = bijection.transform_and_log_det(x, condition)
            return x, log_density - log_det

        ranks = {"x": len(state_shape), "log_density": 0}
        if conditional_shape is not None:
            ranks["condition"] = len(conditional_shape)
        self.apply = autovmap(**ranks)(_apply)

    @property
    def flowjax_bijection(self):
        """Reconstruct the original FlowJAX bijection from stored parameters.

        Returns:
            The FlowJAX bijection with current parameter values.
        """
        return jax.tree.unflatten(self.treedef, self.params)

    def forward(self, x, log_density, **kwargs):
        """Apply forward transformation using FlowJAX bijection.

        Args:
            x: Input to transform.
            log_density: Input log density.
            **kwargs: Additional arguments, including conditional inputs.

        Returns:
            Tuple of (transformed_x, updated_log_density).
        """
        condition = kwargs.get(self.cond_name, None)
        return self.apply(self.flowjax_bijection, x, condition, log_density, False)

    def reverse(self, y, log_density, **kwargs):
        """Apply reverse transformation using FlowJAX bijection.

        Args:
            y: Input to inverse transform.
            log_density: Input log density.
            **kwargs: Additional arguments, including conditional inputs.

        Returns:
            Tuple of (inverse_transformed_y, updated_log_density).
        """
        condition = kwargs.get(self.cond_name, None)
        return self.apply(self.flowjax_bijection, y, condition, log_density, True)


class BijxToFlowjaxBijection(flowjax.bijections.AbstractBijection):
    """Adapter to use bijx bijections with FlowJAX interface.

    Wraps a bijx bijection to implement the FlowJAX AbstractBijection interface.

    Example:
        >>> bijx_bij = bijx.Sigmoid()
        >>> flowjax_bij = BijxToFlowjaxBijection.from_bijection(
        ...     bijx_bij, shape=(5,)
        ... )
        >>> # Now use with FlowJAX
        >>> x = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> y, log_det = flowjax_bij.transform_and_log_det(x)
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    params: nnx.State
    graph: nnx.graph
    cond_name: str = "condition"

    @classmethod
    def from_bijection(
        cls,
        bijection: Bijection,
        shape: tuple[int, ...],
        cond_shape: tuple[int, ...] | None = None,
        cond_name: str = "condition",
    ):
        """Create FlowJAX bijection adapter from bijx bijection.

        Args:
            bijection: The bijx bijection to wrap.
            shape: Event shape of the bijection domain/codomain.
            cond_shape: Shape of conditional inputs, if any.
            cond_name: Name for conditional input parameter.

        Returns:
            BijxToFlowjaxBijection instance wrapping the bijx bijection.
        """
        params, graph = nnx.split(bijection)
        return cls(shape, cond_shape, params, graph, cond_name)

    @property
    def bijx_bijection(self):
        """Reconstruct the original bijx bijection from parameters and graph.

        Returns:
            The bijx bijection with current parameter values.
        """
        return nnx.merge(self.params, self.graph)

    def transform_and_log_det(self, x, condition=None):
        """Apply forward transformation with log determinant.

        Args:
            x: Input to transform.
            condition: Optional conditional input.

        Returns:
            Tuple of (transformed_x, log_determinant) following FlowJAX convention.
        """
        kwargs = {}
        if condition is not None:
            kwargs[self.cond_name] = condition

        # Validate conditional usage
        if condition is not None and self.cond_shape is None:
            raise TypeError(
                "Condition provided but cond_shape is None. "
                "Specify cond_shape when constructing BijxToFlowjaxBijection."
            )
        if condition is not None and self.cond_shape is not None:
            expected = tuple(self.cond_shape)
            if len(expected) > 0:
                actual = tuple(jnp.shape(condition)[-len(expected) :])
            else:
                actual = ()
            if actual != expected:
                raise ValueError(
                    f"condition trailing shape {actual} does not "
                    f"match cond_shape {expected}"
                )

        y, neg_log_det = self.bijx_bijection.forward(x, jnp.zeros(()), **kwargs)
        return y, -neg_log_det

    def inverse_and_log_det(self, y, condition=None):
        """Apply inverse transformation with log determinant.

        Args:
            y: Input to inverse transform.
            condition: Optional conditional input.

        Returns:
            Tuple of (inverse_transformed_y, log_determinant)
            following FlowJAX convention.
        """
        kwargs = {}
        if condition is not None:
            kwargs[self.cond_name] = condition

        # Validate conditional usage
        if condition is not None and self.cond_shape is None:
            raise TypeError(
                "Condition provided but cond_shape is None. "
                "Specify cond_shape when constructing BijxToFlowjaxBijection."
            )
        if condition is not None and self.cond_shape is not None:
            expected = tuple(self.cond_shape)
            if len(expected) > 0:
                actual = tuple(jnp.shape(condition)[-len(expected) :])
            else:
                actual = ()
            if actual != expected:
                raise ValueError(
                    f"condition trailing shape {actual} "
                    f"does not match cond_shape {expected}"
                )

        x, neg_log_det = self.bijx_bijection.reverse(y, jnp.zeros(()), **kwargs)
        return x, -neg_log_det


class FlowjaxToBijxDistribution(Distribution):
    """Adapter to use FlowJAX distributions with bijx interface.

    Wraps a FlowJAX distribution to implement the bijx :class:`Distribution` interface.

    Example:
        >>> import flowjax.distributions as fdist
        >>> flowjax_dist = fdist.Normal(jnp.zeros(3), jnp.ones(3))
        >>> bijx_dist = FlowjaxToBijxDistribution(flowjax_dist)
        >>> samples, log_density = bijx_dist.sample((100,), rng=rngs.next())
    """

    def __init__(
        self, flowjax_dist, cond_name: str = "condition", rngs: nnx.Rngs | None = None
    ):
        """Initialize FlowJAX to bijx distribution adapter.

        Args:
            flowjax_dist: FlowJAX distribution to wrap.
            cond_name: Keyword argument name for passing conditional inputs.
            rngs: Random number generators for bijx compatibility.
        """
        super().__init__(rngs)
        params, self.treedef = jax.tree.flatten(flowjax_dist)
        self.params = nnx.Param(params)
        self.cond_name = cond_name

        self.event_shape = flowjax_dist.shape
        self.conditional_shape = flowjax_dist.cond_shape

    @property
    def flowjax_dist(self):
        """Reconstruct the original FlowJAX distribution from stored parameters.

        Returns:
            The FlowJAX distribution with current parameter values.
        """
        return jax.tree.unflatten(self.treedef, self.params)

    def get_batch_shape(self, x):
        """Infer batch shape from input array and known event shape.

        Args:
            x: Input array to analyze.

        Returns:
            Batch shape tuple inferred from input.
        """
        event_ndim = len(self.event_shape)
        return x.shape[:-event_ndim] if event_ndim > 0 else x.shape

    def sample(self, batch_shape=(), rng=None, **kwargs):
        """Sample from the FlowJAX distribution.

        Args:
            batch_shape: Shape of batch dimensions for samples.
            rng: Random key for sampling.
            **kwargs: Additional arguments, including conditional inputs.

        Returns:
            Tuple of (samples, log_density) following bijx convention.
        """
        rng = self._get_rng(rng)
        condition = kwargs.get(self.cond_name, None)

        flowjax_dist = self.flowjax_dist
        samples = flowjax_dist.sample(rng, batch_shape, condition)
        log_density = flowjax_dist.log_prob(samples, condition)

        return samples, log_density

    def log_density(self, x, **kwargs):
        """Evaluate log density using FlowJAX distribution.

        Args:
            x: Points at which to evaluate log density.
            **kwargs: Additional arguments, including conditional inputs.

        Returns:
            Log density values from the FlowJAX distribution.
        """
        condition = kwargs.get(self.cond_name, None)
        return self.flowjax_dist.log_prob(x, condition)


class BijxToFlowjaxDistribution(flowjax.distributions.AbstractDistribution):
    """Adapter to use bijx distributions with FlowJAX interface.

    Wraps a bijx distribution to implement the FlowJAX AbstractDistribution interface.

    Example:
        >>> bijx_dist = bijx.IndependentNormal(event_shape=(3,))
        >>> flowjax_dist = BijxToFlowjaxDistribution.from_distribution(
        ...     bijx_dist, shape=(3,)
        ... )
        >>> # Now use with FlowJAX
        >>> samples = flowjax_dist.sample(key, (100,))
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    params: nnx.State
    graph: nnx.graph
    cond_name: str = "condition"

    @classmethod
    def from_distribution(
        cls,
        distribution: Distribution,
        shape: tuple[int, ...],
        cond_shape: tuple[int, ...] | None = None,
        cond_name: str = "condition",
    ):
        """Create FlowJAX distribution adapter from bijx distribution.

        Args:
            distribution: The bijx distribution to wrap.
            shape: Event shape of the distribution.
            cond_shape: Shape of conditional inputs, if any.
            cond_name: Name for conditional input parameter.

        Returns:
            BijxToFlowjaxDistribution instance wrapping the bijx distribution.
        """
        params, graph = nnx.split(distribution)
        return cls(shape, cond_shape, params, graph, cond_name)

    @property
    def bijx_dist(self):
        """Reconstruct the original bijx distribution from parameters and graph.

        Returns:
            The bijx distribution with current parameter values.
        """
        return nnx.merge(self.params, self.graph)

    def _sample(self, key, condition=None):
        """Sample from the bijx distribution (FlowJAX interface).

        Args:
            key: Random key for sampling.
            condition: Optional conditional input.

        Returns:
            Single sample from the distribution.
        """
        kwargs = {}
        if condition is not None:
            kwargs[self.cond_name] = condition

        samples, _ = self.bijx_dist.sample(batch_shape=(), rng=key, **kwargs)
        return samples

    def _log_prob(self, x, condition=None):
        """Evaluate log probability using bijx distribution (FlowJAX interface).

        Args:
            x: Points at which to evaluate log probability.
            condition: Optional conditional input.

        Returns:
            Log probability values.
        """
        kwargs = {}
        if condition is not None:
            kwargs[self.cond_name] = condition

        return self.bijx_dist.log_density(x, **kwargs)

    def _sample_and_log_prob(self, key, condition=None):
        """Sample and evaluate log probability jointly (FlowJAX interface).

        Args:
            key: Random key for sampling.
            condition: Optional conditional input.

        Returns:
            Tuple of (sample, log_probability).
        """
        kwargs = {}
        if condition is not None:
            kwargs[self.cond_name] = condition

        samples, log_density = self.bijx_dist.sample(batch_shape=(), rng=key, **kwargs)
        return samples, log_density


def to_flowjax(
    module: Bijection | Distribution,
    shape: tuple[int, ...] | None = None,
    cond_shape: tuple[int, ...] | None = None,
):
    """Convert bijx component to FlowJAX interface.

    Creates FlowJAX-compatible wrappers for bijx bijections and distributions.

    Args:
        module: The bijx bijection or distribution to convert.
        shape: Event shape of the component (required for FlowJAX compatibility).
        cond_shape: Shape of conditional inputs, if any.

    Returns:
        FlowJAX-compatible wrapper implementing the appropriate AbstractBijection
        or AbstractDistribution interface.

    Raises:
        TypeError: If shape is not provided (required for FlowJAX).
        ValueError: If module type is not supported.

    Example:
        >>> bijx_sigmoid = bijx.Sigmoid()
        >>> flowjax_sigmoid = to_flowjax(bijx_sigmoid, shape=(3,))
    """
    if isinstance(module, Bijection):
        if shape is None:
            raise TypeError(
                "Converting bijx bijection to FlowJAX requires 'shape' parameter"
            )
        return BijxToFlowjaxBijection.from_bijection(module, shape, cond_shape)
    elif isinstance(module, Distribution):
        if shape is None:
            raise TypeError(
                "Converting bijx distribution to FlowJAX requires 'shape' parameter"
            )
        return BijxToFlowjaxDistribution.from_distribution(module, shape, cond_shape)
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")


def from_flowjax(
    module: (
        flowjax.bijections.AbstractBijection
        | flowjax.distributions.AbstractDistribution
    ),
):
    """Convert FlowJAX component to bijx interface.

    Creates bijx-compatible wrappers for FlowJAX bijections and distributions.

    Args:
        module: The FlowJAX bijection or distribution to convert.

    Returns:
        bijx-compatible wrapper implementing the appropriate :class:`Bijection`
        or :class:`Distribution` interface.

    Raises:
        ValueError: If module type is not supported.

    Example:
        >>> import flowjax.bijections as fbij
        >>> flowjax_spline = fbij.RationalQuadraticSpline(knots=8, interval=(-1, 1))
        >>> bijx_spline = from_flowjax(flowjax_spline)
        >>> # Now use with bijx interface
        >>> y, log_det = bijx_spline.forward(x, log_density)
    """
    if isinstance(module, flowjax.bijections.AbstractBijection):
        return FlowjaxToBijxBijection(module)
    elif isinstance(module, flowjax.distributions.AbstractDistribution):
        return FlowjaxToBijxDistribution(module)
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
