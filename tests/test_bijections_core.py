"""
Tests for core bijections in bijx.

This module tests:
- All scalar bijections with inverse consistency and log density correctness
- Base bijection classes (Bijection, ApplyBijection, Chain, etc.)
- Shape manipulation bijections (ExpandDims, Reshape, SqueezeDims)
- Binary masks
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from hypothesis import given

# Import all bijections to test
from bijx import (
    AffineLinear,
    ApplyBijection,
    BetaStretch,
    Bijection,
    BinaryMask,
    BufferedSampler,
    Chain,
    CondInverse,
    ExpandDims,
    Exponential,
    Frozen,
    FrozenFilter,
    GaussianCDF,
    IndependentNormal,
    Inverse,
    Partial,
    Power,
    Reshape,
    Scaling,
    ScanChain,
    Shift,
    Sigmoid,
    Sinh,
    SoftPlus,
    SqueezeDims,
    Tan,
    Tanh,
    Transformed,
    checker_mask,
)

from .utils import (
    ATOL,
    RTOL,
    assert_finite_and_real,
    check_inverse,
    check_log_density,
    gaussian_domain_inputs,
    positive_inputs,
    unit_interval_inputs,
)


class TestBijectionBase:
    """Tests for the base Bijection class and fundamental operations."""

    @given(gaussian_domain_inputs())
    def test_identity_bijection(self, x):
        """Test that default Bijection acts as identity over random inputs."""
        bijection = Bijection()
        log_density = jnp.array(0.0)

        y, ld_forward = bijection.forward(x, log_density)
        np.testing.assert_array_equal(y, x)
        np.testing.assert_array_equal(ld_forward, log_density)

        x_back, ld_reverse = bijection.reverse(y, ld_forward)
        np.testing.assert_array_equal(x_back, x)
        np.testing.assert_array_equal(ld_reverse, log_density)

    @given(positive_inputs())
    def test_apply_bijection_wrapper(self, x):
        """Test ApplyBijection base class functionality."""

        class SimpleSquareBijection(ApplyBijection):
            def apply(self, x, log_density, reverse=False, **kwargs):
                if reverse:
                    y = jnp.sqrt(x)
                    log_jac = -jnp.log(2 * jnp.sqrt(x))
                    return y, log_density - log_jac
                else:
                    y = x**2
                    log_jac = jnp.log(2 * jnp.abs(x))
                    return y, log_density - log_jac

        bijection = SimpleSquareBijection()
        check_inverse(bijection, x)


class TestScalarBijections:
    """Comprehensive tests for all scalar bijection types."""

    @given(gaussian_domain_inputs())
    def test_sigmoid_bijection(self, x):
        """Test Sigmoid bijection: unbounded -> [0,1]."""
        bijection = Sigmoid()
        check_inverse(bijection, x)
        check_log_density(bijection, x)

    @given(gaussian_domain_inputs())
    def test_tanh_bijection(self, x):
        """Test Tanh bijection: unbounded -> [-1,1]."""
        bijection = Tanh()
        check_inverse(bijection, x)
        check_log_density(bijection, x)

    @given(unit_interval_inputs())
    def test_tan_bijection(self, x):
        """Test Tan bijection: [0,1] -> unbounded."""
        bijection = Tan()
        check_inverse(bijection, x)
        check_log_density(bijection, x)

    @given(gaussian_domain_inputs())
    def test_exponential_bijection(self, x):
        """Test Exponential bijection: unbounded -> (0,∞)."""
        bijection = Exponential()
        check_inverse(bijection, x)
        check_log_density(bijection, x)

    @given(gaussian_domain_inputs())
    def test_softplus_bijection(self, x):
        """Test SoftPlus bijection: unbounded -> (0,∞)."""
        bijection = SoftPlus()
        check_inverse(bijection, x)
        check_log_density(bijection, x)

    @given(gaussian_domain_inputs())
    def test_sinh_bijection(self, x):
        """Test Sinh bijection: unbounded -> unbounded."""
        bijection = Sinh()
        check_inverse(bijection, x)
        check_log_density(bijection, x)

    @given(x=positive_inputs())
    def test_power_bijection(self, x):
        """Test Power bijection with fixed exponent."""
        bijection = Power(exponent=nnx.Variable(2.5))
        check_inverse(bijection, x)
        check_log_density(bijection, x)

    @given(x=gaussian_domain_inputs())
    def test_gaussian_cdf_bijection(self, x):
        """Test GaussianCDF bijection: unbounded -> [0,1]."""
        bijection = GaussianCDF(rngs=nnx.Rngs(0))  # rngs not used
        check_inverse(bijection, x)
        check_log_density(bijection, x)

    @given(x=unit_interval_inputs())
    def test_beta_stretch_bijection(self, x):
        """Test BetaStretch bijection: [0,1] -> [0,1]."""
        bijection = BetaStretch(a=nnx.Param(jnp.array(2.0)), rngs=nnx.Rngs(0))
        check_inverse(bijection, x)
        check_log_density(bijection, x)

    @given(
        x=gaussian_domain_inputs(),
        scale=gaussian_domain_inputs(shape=()),
        shift=gaussian_domain_inputs(shape=()),
    )
    def test_affine_linear_bijection(self, x, scale, shift):
        """Test AffineLinear bijection: y = ax + b."""
        bijection = AffineLinear(scale=scale, shift=shift)
        check_inverse(bijection, x)
        check_log_density(bijection, x)

    @given(
        x=positive_inputs(shape=(1,)),
        scale=positive_inputs(shape=()),
    )
    def test_scaling_bijection(self, x, scale):
        """Test Scaling bijection: y = ax on positive scalar inputs."""
        bijection = Scaling(scale=scale)
        check_inverse(bijection, x)
        check_log_density(bijection, x)

    @given(
        x=gaussian_domain_inputs(shape=(1,)),
        shift=gaussian_domain_inputs(shape=()),
    )
    def test_shift_bijection(self, x, shift):
        """Test Shift bijection: y = x + b on scalar inputs."""
        bijection = Shift(shift=shift)
        check_inverse(bijection, x)
        check_log_density(bijection, x)


class TestBijectionChaining:
    """Tests for bijection composition and chaining."""

    @given(
        x=gaussian_domain_inputs(),
        params=gaussian_domain_inputs(shape=(3, 2)),
    )
    def test_chain_bijections(self, x, params):
        """Test chaining multiple simple bijections with property-based inputs."""
        # rather trivial, but at least AffineLinear generically don't commute
        bijections = [
            AffineLinear(scale=nnx.Param(scale), shift=nnx.Param(shift))
            for scale, shift in params
        ]

        chain = Chain(*bijections)

        check_inverse(chain, x)
        check_log_density(chain, x)

    @given(
        x=gaussian_domain_inputs(),
        scales=positive_inputs(shape=(12,)),
        shifts=gaussian_domain_inputs(shape=(12,)),
    )
    def test_scan_chain_bijection(self, x, scales, shifts):
        """Test ScanChain applies stacked params sequentially and is invertible."""
        # Build a single bijection whose parameters carry a leading scan dimension
        # Each scan slice corresponds to one step in the sequence
        stack = AffineLinear(scale=nnx.Param(scales), shift=nnx.Param(shifts))
        scan_chain = ScanChain(stack)

        # Core properties
        check_inverse(scan_chain, x)
        check_log_density(scan_chain, x)

        # Equivalence to an explicit Chain of the per-step instances
        graph, variables = nnx.split(stack)
        instances = []
        for t in range(scales.shape[0]):
            vars_t = jax.tree.map(lambda a: a[t], variables)
            instances.append(nnx.merge(graph, vars_t))

        explicit_chain = Chain(
            *(
                AffineLinear(scale=nnx.Param(scales[t]), shift=nnx.Param(shifts[t]))
                for t in range(scales.shape[0])
            )
        )

        y_sc, ld_sc = scan_chain.forward(x, jnp.zeros_like(x))
        y_ch, ld_ch = explicit_chain.forward(x, jnp.zeros_like(x))

        np.testing.assert_allclose(y_sc, y_ch, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld_sc, ld_ch, atol=ATOL, rtol=RTOL)

    @given(x=unit_interval_inputs())
    def test_inverse_bijection(self, x):
        """Test Inverse wrapper with unit-interval inputs."""
        original = Sigmoid()
        inverted = Inverse(original)
        check_inverse(inverted, x)
        check_log_density(inverted, x)

    @given(x=gaussian_domain_inputs())
    def test_frozen_bijection(self, x):
        """Test Frozen bijection that locks parameters using property-based inputs."""
        original = AffineLinear(rngs=nnx.Rngs(0))
        frozen = Frozen(original)

        # assert original has parameters
        _, _, params_original, _ = nnx.split(original, FrozenFilter, nnx.Param, ...)
        assert len(jax.tree.leaves(params_original)) > 0

        # assert frozen one has none
        _, _, params_frozen, _ = nnx.split(frozen, FrozenFilter, nnx.Param, ...)
        assert len(jax.tree.leaves(params_frozen)) == 0

        # check that inverse works
        check_inverse(frozen, x)

    @given(x=unit_interval_inputs(shape=(2,)))
    def test_cond_inverse_runtime_switch(self, x):
        """Test CondInverse switches forward/reverse at runtime."""
        base = Sigmoid()
        ld = jnp.array(0.0)

        ci = CondInverse(base, invert=True)
        y1, ld1 = ci.forward(x, ld)
        y1b, ld1b = base.reverse(x, ld)
        np.testing.assert_allclose(y1, y1b, rtol=RTOL)
        np.testing.assert_allclose(ld1, ld1b, rtol=RTOL)

        ci2 = CondInverse(base, invert=False)
        y2, ld2 = ci2.forward(x, ld)
        y2b, ld2b = base.forward(x, ld)
        np.testing.assert_allclose(y2, y2b, rtol=RTOL)
        np.testing.assert_allclose(ld2, ld2b, rtol=RTOL)


class TestMetaTransformations:
    """Tests for shape manipulation bijections."""

    def test_expand_dims_bijection(self):
        """Test ExpandDims shape transformation."""
        bij = ExpandDims(axis=1)

        x = jnp.arange(4.0).reshape(2, 2)
        log_density = jnp.array(0.0)

        y, ld_forward = bij.forward(x, log_density)
        assert y.shape == (2, 1, 2)  # Expanded at axis=1

        x_back, ld_back = bij.reverse(y, ld_forward)
        np.testing.assert_array_equal(x_back, x)
        np.testing.assert_array_equal(ld_back, log_density)

    def test_squeeze_dims_bijection(self):
        """Test SqueezeDims shape transformation."""
        bij = SqueezeDims(axis=1)

        x = jnp.arange(4.0).reshape(2, 1, 2)
        log_density = jnp.array(0.0)

        y, ld_forward = bij.forward(x, log_density)
        assert y.shape == (2, 2)  # Squeezed at axis=1

        x_back, ld_back = bij.reverse(y, ld_forward)
        np.testing.assert_array_equal(x_back, x)
        np.testing.assert_array_equal(ld_back, log_density)

    def test_reshape_bijection(self):
        """Test Reshape bijection."""
        from_shape = (3, 4)
        to_shape = (4, 3)
        bij = Reshape(from_shape=from_shape, to_shape=to_shape)

        x = jnp.arange(12.0).reshape(3, 4)  # Shape (3, 4)
        log_density = jnp.array(0.0)

        y, ld_forward = bij.forward(x, log_density)
        assert y.shape == to_shape

        x_back, ld_back = bij.reverse(y, ld_forward)
        np.testing.assert_array_equal(x_back, x)
        np.testing.assert_array_equal(ld_back, log_density)

    def test_reshape_with_batch_dimensions(self):
        """Test Reshape with batch dimensions."""
        from_shape = (3, 2)
        to_shape = (2, 3)
        bij = Reshape(from_shape=from_shape, to_shape=to_shape)

        # Batch of samples
        x = jnp.arange(30.0).reshape(5, 3, 2)  # Batch shape (5,), event shape (3, 2)
        log_density = jnp.zeros(5)

        y, ld_forward = bij.forward(x, log_density)
        assert y.shape == (5, 2, 3)  # Batch preserved, event reshaped

        x_back, ld_back = bij.reverse(y, ld_forward)
        np.testing.assert_array_equal(x_back, x)
        np.testing.assert_array_equal(ld_back, log_density)

    def test_partial_passes_and_overrides_kwargs(self):
        """Partial should forward fixed kwargs and allow call-time override."""

        class KwargBijection(Bijection):
            def forward(self, x, log_density, *, delta, factor=1.0):
                return x + factor * delta, log_density

            def reverse(self, x, log_density, *, delta, factor=1.0):
                return x - factor * delta, log_density

        base = KwargBijection()
        part = Partial(base, delta=1.5, factor=2.0)

        x = jnp.array([0.0, 1.0])
        ld = jnp.array(0.0)

        # Uses fixed kwargs when none provided
        y, ld_f = part.forward(x, ld)
        np.testing.assert_allclose(y, x + 3.0, rtol=RTOL)
        np.testing.assert_allclose(ld_f, ld, rtol=RTOL)

        # Call-time override of a single kwarg takes precedence
        y2, _ = part.forward(x, ld, delta=2.0)
        np.testing.assert_allclose(y2, x + 4.0, rtol=RTOL)

        # Override both kwargs
        y3, _ = part.forward(x, ld, delta=1.0, factor=5.0)
        np.testing.assert_allclose(y3, x + 5.0, rtol=RTOL)

        # Reverse should invert with the same kwargs logic
        z, ld_b = part.reverse(y, ld)
        np.testing.assert_allclose(z, x, rtol=RTOL)
        np.testing.assert_allclose(ld_b, ld, rtol=RTOL)


class TestBinaryMask:
    """Tests for BinaryMask and coupling layer utilities."""

    def test_binary_mask_split_merge_1d(self):
        """Test BinaryMask split/merge for 1D arrays."""
        mask = BinaryMask.from_boolean_mask(jnp.array([True, False, True, False]))

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        primary, secondary = mask.split(x)

        # Primary should contain elements at True positions
        expected_primary = jnp.array([1.0, 3.0])
        expected_secondary = jnp.array([2.0, 4.0])

        np.testing.assert_array_equal(primary, expected_primary)
        np.testing.assert_array_equal(secondary, expected_secondary)

        # Test round-trip
        reconstructed = mask.merge(primary, secondary)
        np.testing.assert_array_equal(x, reconstructed)

    def test_binary_mask_split_merge_2d(self):
        """Test BinaryMask split/merge for 2D arrays."""
        mask = checker_mask((2, 2), parity=True)

        x = jnp.arange(4.0).reshape(2, 2)
        primary, secondary = mask.split(x)
        reconstructed = mask.merge(primary, secondary)

        np.testing.assert_array_equal(x, reconstructed)

    def test_binary_mask_with_batches(self):
        """Test BinaryMask with batch dimensions."""
        mask = checker_mask((2, 2), parity=False)

        x_batch = jnp.arange(8.0).reshape(2, 2, 2)

        primary_batch, secondary_batch = mask.split(x_batch)
        reconstructed_batch = mask.merge(primary_batch, secondary_batch)

        np.testing.assert_array_equal(x_batch, reconstructed_batch)

    def test_binary_mask_flip(self):
        """Test BinaryMask flip operation."""
        mask = checker_mask((2, 2), parity=True)
        mask_inv = mask.flip()
        mask_inv2 = ~mask  # Test operator overload

        x = jnp.arange(4.0).reshape(2, 2)

        # Primary/secondary should be swapped
        p1, s1 = mask.split(x)
        p2, s2 = mask_inv.split(x)

        np.testing.assert_array_equal(p1, s2)
        np.testing.assert_array_equal(s1, p2)

        # Test operator overload gives same result
        p3, s3 = mask_inv2.split(x)
        np.testing.assert_array_equal(p2, p3)
        np.testing.assert_array_equal(s2, s3)

    def test_checker_mask_patterns(self):
        """Test checker_mask creates correct patterns."""
        mask_even = checker_mask((3, 3), parity=False)
        mask_odd = checker_mask((3, 3), parity=True)

        # Patterns should be inverted
        bool_even = mask_even.boolean_mask
        bool_odd = mask_odd.boolean_mask

        np.testing.assert_array_equal(bool_even, ~bool_odd)

        # Check that the pattern alternates
        expected_even = jnp.array(
            [[False, True, False], [True, False, True], [False, True, False]]
        )

        np.testing.assert_array_equal(bool_even, expected_even)


class TestBijectionIntegration:
    """Integration tests across different bijection types."""

    def test_complex_bijection_chain(self):
        """Test a chain combining different bijection types."""
        r = nnx.Rngs(jax.random.key(0))
        chain = Chain(
            Shift(rngs=r),
            Sigmoid(),
            BetaStretch(a=nnx.Param(jnp.array(2.0)), rngs=r),
            Tanh(),
        )

        x = jnp.array([0.0])
        check_inverse(chain, x)
        check_log_density(chain, x)

    def test_bijection_gradient_flow(self, rng_key):
        """Test that gradients can be computed."""

        def loss_fn(params):
            bijection = AffineLinear(rngs=nnx.Rngs(rng_key))
            # Manually set parameters for testing via the underlying param objects
            bijection.scale.param.value = params["scale"]
            bijection.shift.param.value = params["shift"]

            x = jnp.array([1.0, 2.0])
            log_density = jnp.array(0.0)
            y, ld = bijection.forward(x, log_density)
            return jnp.sum(y**2) - ld  # dummy loss function

        params = {"scale": jnp.array([1.5, 2.0]), "shift": jnp.array([0.1, -0.2])}

        loss_val, grads = jax.value_and_grad(loss_fn)(params)

        assert_finite_and_real(jnp.array(loss_val), "loss")
        for key, grad in grads.items():
            assert_finite_and_real(grad, f"gradient_{key}")
            assert grad.shape == params[key].shape


class TestCompositeSamplers:
    """Tests for composite sampling functionality."""

    def test_transformed_sampler_basic(self, rng_key):
        """Test basic Transformed sampler functionality."""
        # Base distribution
        prior = IndependentNormal(event_shape=(2,))

        # Bijection
        bijection = Sigmoid()

        # Transformed distribution
        transformed = Transformed(prior, bijection)

        # Test sampling
        rng = rng_key
        samples, log_density = transformed.sample((10,), rng=rng)

        assert samples.shape == (10, 2)
        assert log_density.shape == (10,)
        assert_finite_and_real(samples, "transformed samples")
        assert_finite_and_real(log_density, "transformed log density")

        # Samples should be in unit interval (due to Sigmoid)
        assert jnp.all((samples >= 0) & (samples <= 1))

        # Test log density evaluation
        test_x = jnp.array([[0.3, 0.7], [0.1, 0.9]])
        log_prob = transformed.log_density(test_x)

        assert log_prob.shape == (2,)
        assert_finite_and_real(log_prob, "transformed log prob")

    def test_buffered_sampler_basic(self, rng_key):
        """Test basic BufferedSampler functionality."""
        base_dist = IndependentNormal(
            event_shape=(3,), rngs=nnx.Rngs(rng_key)
        )  # Provide rngs before creating buffer

        # Create buffered sampler
        buffered = BufferedSampler(base_dist, buffer_size=5)

        # Test sampling
        rng = rng_key
        samples, log_density = buffered.sample((5,), rng=rng)

        assert samples.shape == (5, 3)
        assert log_density.shape == (5,)
        assert_finite_and_real(samples, "buffered samples")
        assert_finite_and_real(log_density, "buffered log density")

        old_sample, _ = buffered.sample(rng=rng)
        assert old_sample.shape == (3,)

        # make sure samples are not repeated
        for _ in range(10):
            new_sample, _ = buffered.sample(rng=rng)
            assert new_sample.shape == (3,)
            assert not jnp.allclose(new_sample, old_sample)
            old_sample = new_sample
