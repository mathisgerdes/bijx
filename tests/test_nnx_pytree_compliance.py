"""Tests for NNX pytree compliance.

Starting with Flax 0.12.0, NNX modules must not have raw arrays in static attributes.
This test suite ensures all bijections and distributions properly wrap state objects
with nnx.data() or other appropriate containers.
"""

import diffrax
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import bijx


def simple_vector_field(t, x):
    """Simple vector field for testing: linear decay."""
    return -x, jnp.sum(x, axis=-1, keepdims=True)


class SimpleVectorFieldModule(nnx.Module):
    """Parametric vector field module for testing."""

    def __init__(self, dim, rngs):
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (dim, dim)))
        self.bias = nnx.Param(jax.random.normal(rngs.params(), (dim,)))

    def __call__(self, t, x):
        """Linear vector field with learnable parameters."""
        dx_dt = x @ self.weight.value.T + self.bias.value
        # Divergence of linear transformation is trace of weight matrix
        div = jnp.trace(self.weight.value)
        return dx_dt, -div


def check_pytree_compliance(module):
    """Check that a module is pytree-compliant according to Flax 0.12.0 rules.

    This function instantiates the module and verifies it doesn't raise
    a ValueError about unexpected Arrays in static attributes.

    Args:
        module: An NNX module instance to check.

    Raises:
        AssertionError: If the module is not pytree-compliant.
    """
    # The check happens during __init__, so if we get here without error,
    # the module is compliant. We'll also do a split/merge cycle to be sure.
    try:
        # If we can split and merge, the module is pytree-compliant
        graph, state = nnx.split(module)
        reconstructed = nnx.merge(graph, state)
        return reconstructed
    except ValueError as e:
        if "unexpected Arrays" in str(e):
            pytest.fail(f"Module {type(module).__name__} is not pytree-compliant: {e}")
        raise


class TestContFlowDiffraxPytreeCompliance:
    """Test ContFlowDiffrax for NNX pytree compliance."""

    def test_contflow_diffrax_with_module_vf_instantiation(self):
        """Test that ContFlowDiffrax can be instantiated with a parametric vector field.

        This test specifically checks the scenario reported in the bug:
        when ContFlowDiffrax wraps a parametric vector field module,
        it should properly wrap the vf_variables state with nnx.data().
        """
        rngs = nnx.Rngs(0)
        dim = 4

        # Create a parametric vector field
        vf = SimpleVectorFieldModule(dim, rngs)

        # This should not raise ValueError about unexpected Arrays
        flow = bijx.bijections.ContFlowDiffrax(
            vf=vf,
            config=bijx.DiffraxConfig(
                solver=diffrax.Tsit5(),
                dt=0.1,
                t_start=0.0,
                t_end=1.0,
            ),
        )

        check_pytree_compliance(flow)

    def test_contflow_diffrax_split_merge_cycle(self):
        """Test that ContFlowDiffrax can be split and merged without issues."""
        rngs = nnx.Rngs(0)
        dim = 3

        vf = SimpleVectorFieldModule(dim, rngs)
        flow = bijx.bijections.ContFlowDiffrax(vf=vf)

        # Split and merge should work without errors
        graph, state = nnx.split(flow)
        reconstructed = nnx.merge(graph, state)

        # Verify the reconstructed flow works
        x = jax.random.normal(rngs.params(), (5, dim))
        log_density = jnp.zeros(5)

        y1, ld1 = flow.forward(x, log_density)
        y2, ld2 = reconstructed.forward(x, log_density)

        assert jnp.allclose(y1, y2)
        assert jnp.allclose(ld1, ld2)

    def test_contflow_diffrax_jit_compilation(self):
        """Test that ContFlowDiffrax can be JIT compiled.

        This is an important use case - if the module isn't pytree-compliant,
        JIT compilation might fail or behave unexpectedly.
        """
        rngs = nnx.Rngs(0)
        dim = 3

        vf = SimpleVectorFieldModule(dim, rngs)
        flow = bijx.bijections.ContFlowDiffrax(vf=vf)

        @nnx.jit
        def forward_fn(flow, x, log_density):
            return flow.forward(x, log_density)

        x = jax.random.normal(rngs.params(), (5, dim))
        log_density = jnp.zeros(5)

        # Should compile and run without errors
        y, ld = forward_fn(flow, x, log_density)

        assert y.shape == x.shape
        assert ld.shape == log_density.shape


class TestMixtureStackPytreeCompliance:
    """Test MixtureStack for NNX pytree compliance.

    MixtureStack also uses nnx.split to extract distribution variables,
    and should properly wrap them with nnx.data().
    """

    def test_mixture_stack_instantiation(self):
        """Test that MixtureStack properly wraps distribution variables."""
        rngs = nnx.Rngs(0)

        # Create a stack of distributions (using vmap to create batched parameters)
        means = jax.random.normal(rngs.params(), (5, 3))  # 5 components, 3D each
        scales = jnp.ones((5, 3))

        # Use vmap to create a stack
        dist_stack = nnx.vmap(
            lambda mean, scale: bijx.DiagonalNormal(
                mean=nnx.Param(mean), scales=nnx.Param(scale)
            )
        )(nnx.Param(means), nnx.Param(scales))

        # This should not raise ValueError about unexpected Arrays
        mixture = bijx.MixtureStack(dist_stack, weights=(5,), rngs=rngs)

        check_pytree_compliance(mixture)

    def test_mixture_stack_split_merge(self):
        """Test that MixtureStack can be split and merged."""
        rngs = nnx.Rngs(0)

        means = jax.random.normal(rngs.params(), (3, 2))  # 3 components, 2D each
        scales = jnp.ones((3, 2))

        dist_stack = nnx.vmap(
            lambda mean, scale: bijx.DiagonalNormal(
                mean=nnx.Param(mean), scales=nnx.Param(scale)
            )
        )(nnx.Param(means), nnx.Param(scales))

        mixture = bijx.MixtureStack(dist_stack, weights=(3,), rngs=rngs)

        # Split and merge should work
        graph, state = nnx.split(mixture)
        reconstructed = nnx.merge(graph, state)

        # Verify functionality is preserved
        x = jax.random.normal(rngs.params(), (10, 2))
        ld1 = mixture.log_density(x)
        ld2 = reconstructed.log_density(x)

        assert jnp.allclose(ld1, ld2)


class TestGeneralPytreeCompliance:
    """General tests for pytree compliance across all bijections and distributions."""

    @pytest.mark.parametrize(
        "bijection_factory",
        [
            lambda rngs: bijx.bijections.Tanh(),
            lambda rngs: bijx.bijections.Sigmoid(),
            lambda rngs: bijx.bijections.Exponential(),
            lambda rngs: bijx.bijections.Power(rngs=rngs),
            lambda rngs: bijx.bijections.AffineLinear(rngs=rngs),
        ],
        ids=["Tanh", "Sigmoid", "Exponential", "Power", "AffineLinear"],
    )
    def test_scalar_bijections_compliance(self, bijection_factory):
        """Test that scalar bijections are pytree-compliant."""
        rngs = nnx.Rngs(0)
        bijection = bijection_factory(rngs)
        check_pytree_compliance(bijection)

    @pytest.mark.parametrize(
        "distribution_factory",
        [
            lambda rngs: bijx.IndependentNormal((3, 4), rngs=rngs),
            lambda rngs: bijx.DiagonalNormal((5,), rngs=rngs),
            lambda rngs: bijx.MultivariateNormal.given_dim(4, rngs=rngs),
        ],
        ids=["IndependentNormal", "DiagonalNormal", "MultivariateNormal"],
    )
    def test_distributions_compliance(self, distribution_factory):
        """Test that distributions are pytree-compliant."""
        rngs = nnx.Rngs(0)
        distribution = distribution_factory(rngs)
        check_pytree_compliance(distribution)

    def test_chain_bijection_compliance(self):
        """Test that Chain bijections are pytree-compliant."""
        rngs = nnx.Rngs(0)

        chain = bijx.bijections.Chain(
            bijx.bijections.Tanh(),
            bijx.bijections.AffineLinear(rngs=rngs),
            bijx.bijections.Sigmoid(),
        )

        check_pytree_compliance(chain)

    def test_scan_chain_compliance(self):
        """Test that ScanChain bijections are pytree-compliant."""
        # Create a stack of parameter-free transformations
        n_layers = 5

        stack = nnx.vmap(lambda _: bijx.bijections.Tanh())(jnp.arange(n_layers))

        scan_chain = bijx.bijections.ScanChain(stack)

        check_pytree_compliance(scan_chain)


class TestContFlowRK4Compliance:
    """Test ContFlowRK4 for pytree compliance.

    ContFlowRK4 stores a callable vector field, which should be fine
    as long as it doesn't contain unwrapped arrays.
    """

    def test_contflow_rk4_with_function(self):
        """Test ContFlowRK4 with a simple function vector field."""
        flow = bijx.bijections.ContFlowRK4(simple_vector_field, steps=20)
        check_pytree_compliance(flow)

    def test_contflow_rk4_with_module(self):
        """Test ContFlowRK4 with a module vector field."""
        rngs = nnx.Rngs(0)
        dim = 3

        vf = SimpleVectorFieldModule(dim, rngs)
        flow = bijx.bijections.ContFlowRK4(vf, steps=20)

        check_pytree_compliance(flow)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
