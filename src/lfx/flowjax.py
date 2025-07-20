import flowjax
import flowjax.bijections
import flowjax.distributions
import jax
import jax.numpy as jnp
from flax import nnx
from jax_autovmap import auto_vmap

from .bijections.base import Bijection
from .distributions import Distribution


class FlowjaxToLfxBijection(Bijection):
    """Wrap a flowjax bijection to work with LFX interface."""

    def __init__(self, flowjax_bijection, cond_name: str = "condition"):
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
        self.apply = auto_vmap(**ranks)(_apply)

    @property
    def flowjax_bijection(self):
        return jax.tree.unflatten(self.treedef, self.params)

    def forward(self, x, log_density, **kwargs):
        condition = kwargs.get(self.cond_name, None)
        return self.apply(self.flowjax_bijection, x, condition, log_density, False)

    def reverse(self, y, log_density, **kwargs):
        condition = kwargs.get(self.cond_name, None)
        return self.apply(self.flowjax_bijection, y, condition, log_density, True)


class LfxToFlowjaxBijection(flowjax.bijections.AbstractBijection):
    """Wrap an LFX bijection to work with flowjax interface."""

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
        params, graph = nnx.split(bijection)
        return cls(shape, cond_shape, params, graph, cond_name)

    @property
    def lfx_bijection(self):
        return nnx.merge(self.params, self.graph)

    def transform_and_log_det(self, x, condition=None):
        kwargs = {}
        if condition is not None:
            kwargs[self.cond_name] = condition

        y, neg_log_det = self.lfx_bijection.forward(x, jnp.zeros(()), **kwargs)
        return y, -neg_log_det

    def inverse_and_log_det(self, y, condition=None):
        kwargs = {}
        if condition is not None:
            kwargs[self.cond_name] = condition

        x, neg_log_det = self.lfx_bijection.reverse(y, jnp.zeros(()), **kwargs)
        return x, -neg_log_det


class FlowjaxToLfxDistribution(Distribution):
    """Wrap a flowjax distribution to work with LFX interface."""

    def __init__(
        self, flowjax_dist, cond_name: str = "condition", rngs: nnx.Rngs | None = None
    ):
        super().__init__(rngs)
        params, self.treedef = jax.tree.flatten(flowjax_dist)
        self.params = nnx.Param(params)
        self.cond_name = cond_name

        self.event_shape = flowjax_dist.shape
        self.conditional_shape = flowjax_dist.cond_shape

    @property
    def flowjax_dist(self):
        return jax.tree.unflatten(self.treedef, self.params)

    def get_batch_shape(self, x):
        event_ndim = len(self.event_shape)
        return x.shape[:-event_ndim] if event_ndim > 0 else x.shape

    def sample(self, batch_shape=(), rng=None, **kwargs):
        rng = self._get_rng(rng)
        condition = kwargs.get(self.cond_name, None)

        flowjax_dist = self.flowjax_dist
        samples = flowjax_dist.sample(rng, batch_shape, condition)
        log_density = flowjax_dist.log_prob(samples, condition)

        return samples, log_density

    def log_density(self, x, **kwargs):
        condition = kwargs.get(self.cond_name, None)
        return self.flowjax_dist.log_prob(x, condition)


class LfxToFlowjaxDistribution(flowjax.distributions.AbstractDistribution):
    """Wrap an LFX distribution to work with flowjax interface."""

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
        params, graph = nnx.split(distribution)
        return cls(shape, cond_shape, params, graph, cond_name)

    @property
    def lfx_dist(self):
        return nnx.merge(self.params, self.graph)

    def _sample(self, key, condition=None):
        kwargs = {}
        if condition is not None:
            kwargs[self.cond_name] = condition

        samples, _ = self.lfx_dist.sample(batch_shape=(), rng=key, **kwargs)
        return samples

    def _log_prob(self, x, condition=None):
        kwargs = {}
        if condition is not None:
            kwargs[self.cond_name] = condition

        return self.lfx_dist.log_density(x, **kwargs)

    def _sample_and_log_prob(self, key, condition=None):
        kwargs = {}
        if condition is not None:
            kwargs[self.cond_name] = condition

        samples, log_density = self.lfx_dist.sample(batch_shape=(), rng=key, **kwargs)
        return samples, log_density


def to_flowjax(
    module: Bijection | Distribution,
    shape: tuple[int, ...] | None = None,
    cond_shape: tuple[int, ...] | None = None,
):
    if isinstance(module, Bijection):
        if shape is None:
            raise TypeError(
                "Converting LFX bijection to FlowJAX requires 'shape' parameter"
            )
        return LfxToFlowjaxBijection.from_bijection(module, shape, cond_shape)
    elif isinstance(module, Distribution):
        if shape is None:
            raise TypeError(
                "Converting LFX distribution to FlowJAX requires 'shape' parameter"
            )
        return LfxToFlowjaxDistribution.from_distribution(module, shape, cond_shape)
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")


def from_flowjax(
    module: (
        flowjax.bijections.AbstractBijection
        | flowjax.distributions.AbstractDistribution
    ),
):
    if isinstance(module, flowjax.bijections.AbstractBijection):
        return FlowjaxToLfxBijection(module)
    elif isinstance(module, flowjax.distributions.AbstractDistribution):
        return FlowjaxToLfxDistribution(module)
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
