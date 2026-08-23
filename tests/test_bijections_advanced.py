"""
Tests for advanced bijections in bijx.

This module tests:
- Continuous normalizing flows (ContFlowDiffrax, ContFlowRK4, ContFlowCG)
- Automatic Jacobian vector fields (AutoJacVF)
- Rational quadratic splines (MonotoneRQSpline)
- Fourier space bijections (SpectrumScaling, FreeTheoryScaling)
- Advanced coupling layers with sophisticated parameter management
"""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from hypothesis import given

# Import bijections to test
from bijx import (
    AffineLinear,
    AutoJacVF,
    Chain,
    ContFlowDiffrax,
    ContFlowRK4,
    ConvVF,
    DiffraxConfig,
    FreeTheoryScaling,
    ModuleReconstructor,
    MonotoneRQSpline,
    SpectrumScaling,
    ToFourierData,
    rational_quadratic_spline,
)
from bijx.fourier import FourierMeta, fft_momenta, spectrum_asymmetry

# Import test utilities
from .utils import (
    ATOL,
    ATOL_RELAXED,
    RTOL,
    RTOL_RELAXED,
    assert_finite_and_real,
    check_inverse,
    check_log_density,
    gaussian_domain_inputs,
    unit_interval_inputs,
)


class TestContinuousFlows:
    """Tests for continuous normalizing flows."""

    @given(x=gaussian_domain_inputs(shape=()))
    def test_auto_jac_vf_scalar(self, x):
        """Test AutoJacVF for scalar vector fields (randomized x)."""

        def simple_vf(t, x):
            # Simple quadratic vector field: dx/dt = x^2 - 1
            return x**2 - 1.0

        # Wrap with automatic Jacobian computation
        auto_vf = AutoJacVF(simple_vf, event_dim=0)

        t = 0.5

        dx_dt, dlogp_dt = auto_vf(t, x)

        # Check shapes
        assert dx_dt.shape == x.shape
        assert dlogp_dt.shape == ()

        # sanity check output is what we defined it to be
        np.testing.assert_array_equal(dx_dt, x**2 - 1.0)

        # Check that dlogp_dt is -divergence (for scalar: -d/dx of dx_dt)
        # d/dx(x^2 - 1) = 2x, so -divergence = -2x
        expected_dlogp_dt = -2 * x
        np.testing.assert_allclose(dlogp_dt, expected_dlogp_dt, rtol=RTOL, atol=ATOL)

    @given(x=gaussian_domain_inputs(shape=(2,)))
    def test_auto_jac_vf_vector(self, x):
        """Test AutoJacVF for vector fields (randomized x)."""

        def spiral_vf(t, x):
            # 2D spiral vector field
            x1, x2 = x[..., 0], x[..., 1]
            dx1_dt = -x2 + 0.1 * x1
            dx2_dt = x1 + 0.1 * x2
            return jnp.stack([dx1_dt, dx2_dt], axis=-1)

        # Wrap with automatic Jacobian computation for vector field
        auto_vf = AutoJacVF(spiral_vf, event_dim=1)

        t = 0.0

        dx_dt, dlogp_dt = auto_vf(t, x)

        # Check shapes
        assert dx_dt.shape == x.shape
        assert dlogp_dt.shape == ()

        # Check that dx_dt matches the vector field definition
        expected_dx_dt = jnp.stack(
            [-x[..., 1] + 0.1 * x[..., 0], x[..., 0] + 0.1 * x[..., 1]], axis=-1
        )
        np.testing.assert_allclose(dx_dt, expected_dx_dt, rtol=RTOL, atol=ATOL)

        # For this particular spiral field, divergence = 0.1 + 0.1 = 0.2
        # So -divergence = -0.2
        expected_dlogp_dt = -0.2
        np.testing.assert_allclose(dlogp_dt, expected_dlogp_dt, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize(
        "make_flow",
        [
            lambda vf: ContFlowRK4(vf, t_end=1.0, steps=10),
            lambda vf: ContFlowDiffrax(
                vf,
                DiffraxConfig(
                    t_start=0.0,
                    t_end=1.0,
                    dt=0.1,
                    solver=diffrax.Heun(),
                    stepsize_controller=diffrax.ConstantStepSize(),
                ),
            ),
        ],
        ids=["rk4", "diffrax"],
    )
    def test_cont_flow_roundtrip(self, make_flow):
        """ContFlowRK4 / ContFlowDiffrax forward/reverse round-trip on a linear VF."""

        class LinearVF(nnx.Module):
            """Simple linear vector field: dx/dt = -0.2 x (div = -0.2 -> -div = 0.2)."""

            def __call__(self, t, x, **kwargs):
                return -0.2 * x, jnp.array(0.2)

        flow = make_flow(LinearVF())

        x = jnp.array([1.5])
        log_density = jnp.array(0.0)

        y, ld_forward = flow.forward(x, log_density)
        assert_finite_and_real(y, "cont-flow forward output")
        assert_finite_and_real(ld_forward, "cont-flow forward log density")

        x_back, ld_back = flow.reverse(y, ld_forward)

        # Relaxed tolerance for numerical integration round-trip
        np.testing.assert_allclose(x_back, x, atol=ATOL_RELAXED, rtol=RTOL_RELAXED)
        np.testing.assert_allclose(
            ld_back, log_density, atol=ATOL_RELAXED, rtol=RTOL_RELAXED
        )

    def test_conv_cnf_build_and_call(self, rng_key):
        """ConvCNF build test: shape handling and divergence shape."""
        cnf = ConvVF.build(
            kernel_shape=(3, 3), channel_shape=(1,), rngs=nnx.Rngs(rng_key)
        )
        x = jnp.ones((4, 4, 1))
        t = jnp.array(0.0)
        vel, neg_div = cnf(t, x)
        assert vel.shape == x.shape
        assert neg_div.shape == ()


class TestSplines:
    """Tests for rational quadratic splines."""

    def test_rational_quadratic_spline_basic(self):
        """Test basic rational quadratic spline functionality."""
        # Simple test with known parameters
        inputs = jnp.array([0.3, 0.7])

        # Create uniform bins (4 bins)
        n_bins = 4
        n_knots = n_bins - 1  # internal knots

        bin_widths = jnp.ones((2, n_bins))
        bin_heights = jnp.ones((2, n_bins))
        knot_slopes = jnp.ones((2, n_knots))

        # Test forward transformation
        outputs, log_det = rational_quadratic_spline(
            inputs, bin_widths, bin_heights, knot_slopes
        )

        assert_finite_and_real(outputs, "spline outputs")
        assert_finite_and_real(log_det, "spline log det")
        assert outputs.shape == inputs.shape
        assert log_det.shape == inputs.shape

        # Test inverse transformation
        inputs_back, log_det_inv = rational_quadratic_spline(
            outputs, bin_widths, bin_heights, knot_slopes, inverse=True
        )

        # Check inverse consistency
        np.testing.assert_allclose(inputs_back, inputs, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(log_det + log_det_inv, 0.0, atol=ATOL, rtol=RTOL)

    def test_spline_numerical_stability_boundaries(self):
        """Check stability for inputs near 0 and 1 and for extreme parameters."""
        eps = 1e-12
        x = jnp.array([0.0, eps, 1.0 - eps, 1.0])
        n_bins = 5
        n_knots = n_bins - 1
        big = 40.0
        small = -40.0
        widths = jnp.array([[big] * n_bins])
        heights = jnp.array([[small] * n_bins])
        slopes = jnp.array([[big] * n_knots])

        y, ld = rational_quadratic_spline(x, widths, heights, slopes)
        assert_finite_and_real(y, "spline forward near-boundary outputs")
        assert_finite_and_real(ld, "spline forward near-boundary log det")

        x2, ld2 = rational_quadratic_spline(y, widths, heights, slopes, inverse=True)
        assert_finite_and_real(x2, "spline inverse near-boundary outputs")
        assert_finite_and_real(ld2, "spline inverse near-boundary log det")

        np.testing.assert_allclose(x2, jnp.clip(x, 0, 1), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld + ld2, 0.0, atol=ATOL, rtol=RTOL)

    # Note: vectorized randomized param round-trip checked elsewhere

    def test_monotone_rq_spline_bijection(self, rng_key):
        """Test MonotoneRQSpline as a bijection."""
        # Create spline bijection
        n_knots = 6
        event_shape = ()  # scalar
        spline = MonotoneRQSpline(n_knots, event_shape, rngs=nnx.Rngs(rng_key))

        # Test with unit interval inputs
        x = jnp.array([0.2, 0.5, 0.8])
        log_density = jnp.zeros(3)

        # Check forward transformation
        y, ld_forward = spline.forward(x, log_density)
        assert_finite_and_real(y, "spline forward output")
        assert_finite_and_real(ld_forward, "spline forward log density")

        # Check inverse transformation
        x_back, ld_back = spline.reverse(y, ld_forward)

        # Check inverse consistency
        np.testing.assert_allclose(x_back, x, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld_back, log_density, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    @pytest.mark.parametrize("event_shape", [(), (1,), (2,), (2, 2)])
    @pytest.mark.parametrize("knots", [4, 7])
    def test_monotone_rq_spline_vectorized_roundtrip(
        self, batch_shape, event_shape, knots, rng_key
    ):
        """Round-trip test across batch/event shape variants"""
        rngs = nnx.Rngs(rng_key)
        spline = MonotoneRQSpline(knots, event_shape, rngs=rngs)

        # Unit-interval inputs of shape batch + event
        if event_shape == ():
            x = jax.random.uniform(
                rngs(),
                shape=batch_shape or (),
                minval=0.05,
                maxval=0.95,
            )
            ld = jnp.zeros(batch_shape or ())
        else:
            x = jax.random.uniform(
                rngs(),
                shape=batch_shape + event_shape,
                minval=0.05,
                maxval=0.95,
            )
            ld = jnp.zeros(batch_shape)

        y, ld1 = spline.forward(x, ld)
        x2, ld2 = spline.reverse(y, ld1)

        assert_finite_and_real(y, "vectorized spline outputs")
        assert_finite_and_real(ld1, "vectorized spline log det forward")
        assert_finite_and_real(x2, "vectorized spline inverse outputs")
        assert_finite_and_real(ld2, "vectorized spline log det inverse")

        np.testing.assert_allclose(x2, x, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld2, ld, atol=ATOL, rtol=RTOL)

    @given(x=unit_interval_inputs(shape=()))
    def test_spline_property_based(self, x):
        """Property-based test for spline consistency over random domain inputs."""
        spline = MonotoneRQSpline(4, (), rngs=nnx.Rngs(0))  # Small spline for speed

        # Use safe checks for property-based testing with diagnostics
        check_inverse(spline, x)
        check_log_density(spline, x)


class TestFourierBijections:
    """Tests for Fourier space bijections."""

    def test_spectrum_scaling_basic(self):
        """Test basic SpectrumScaling functionality."""
        # Create simple 2D field for testing
        field_shape = (4, 4)
        x = jnp.ones(field_shape)
        log_density = jnp.array(0.0)

        # Create momentum-dependent scaling
        k_grid = fft_momenta(field_shape)
        k_squared = jnp.sum(k_grid**2, axis=-1)
        scaling = jnp.exp(-0.1 * k_squared)  # Exponential damping

        bijection = SpectrumScaling(scaling, channel_dim=0)

        # Test forward transformation
        y, ld_forward = bijection.forward(x, log_density)
        assert y.shape == x.shape
        assert_finite_and_real(y, "spectrum scaling output")
        assert_finite_and_real(ld_forward, "spectrum scaling log density")

        # Test inverse transformation
        x_back, ld_back = bijection.reverse(y, ld_forward)

        # Check inverse consistency
        np.testing.assert_allclose(x_back, x, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld_back, log_density, atol=ATOL, rtol=RTOL)

    def test_free_theory_scaling(self, rng_key):
        """Test FreeTheoryScaling for physics applications."""
        # Create lattice field (small: this checks round-trip correctness, not scaling)
        lattice_shape = (4, 4)
        x = jax.random.normal(rng_key, lattice_shape)
        log_density = jnp.array(0.0)

        # Create free theory scaling (mass term)
        mass_squared = 0.5
        bijection = FreeTheoryScaling(mass_squared, lattice_shape, channel_dim=0)

        # Test basic functionality
        y, ld_forward = bijection.forward(x, log_density)
        assert y.shape == x.shape
        assert_finite_and_real(y, "free theory output")
        assert_finite_and_real(ld_forward, "free theory log density")

        # Test inverse transformation
        x_back, ld_back = bijection.reverse(y, ld_forward)

        # Check inverse consistency
        np.testing.assert_allclose(x_back, x, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld_back, log_density, atol=ATOL, rtol=RTOL)

    def test_to_fourier_data_bijection(self, rng_key):
        """Test ToFourierData conversion bijection."""
        # Create real field
        field_shape = (6, 6)
        x = jax.random.normal(rng_key, field_shape)
        log_density = jnp.array(0.0)

        from bijx.fourier import FFTRep

        bijection = ToFourierData(field_shape, rep=FFTRep.rfft)

        # Test forward (real -> Fourier)
        y, ld_forward = bijection.forward(x, log_density)
        # Output is FourierData object, check its .data attribute
        assert hasattr(y, "data"), "ToFourierData should return FourierData object"
        # FFT output is complex, so just check it's finite
        assert not jnp.any(
            jnp.isnan(y.data)
        ), "ToFourierData forward output contains NaN"
        assert not jnp.any(
            jnp.isinf(y.data)
        ), "ToFourierData forward output contains inf"
        assert_finite_and_real(ld_forward, "ToFourierData forward log density")

        # Test inverse (Fourier -> real)
        x_back, ld_back = bijection.reverse(y, ld_forward)

        # Check inverse consistency
        np.testing.assert_allclose(x_back, x, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld_back, log_density, atol=ATOL, rtol=RTOL)

    # -- spectrum scaling with channel dimensions --------------------------
    #
    # The spectrum must be consistent on the conjugate pairs stored in the
    # rFFT grid (see test_spectrum_scaling_pair_precondition below), otherwise
    # the rFFT-space multiplication is not an invertible map on real fields.
    # Any real function of ``fft_momenta`` satisfies that, since those momenta
    # are folded into the first Brillouin zone.

    @staticmethod
    def _symmetric_spectrum(space_shape):
        k = fft_momenta(space_shape, lattice=True)
        return jnp.exp(-0.15 * jnp.sum(k**2, axis=-1)) + 0.3

    @staticmethod
    def _dense_log_det(bijection, x):
        """log|det J| from the dense Jacobian of the flattened forward map."""

        def flat_forward(flat):
            y, _ = bijection.forward(flat.reshape(x.shape), jnp.zeros(()))
            return y.ravel()

        jac = jax.jacfwd(flat_forward)(x.ravel())
        return jnp.linalg.slogdet(jac)[1]

    @pytest.mark.parametrize("space_shape", [(8,), (4, 4)])
    @pytest.mark.parametrize("channels", [None, 1, 3])
    def test_spectrum_scaling_log_det_vs_jacobian(self, rng_key, space_shape, channels):
        """Reported log-density change matches slogdet of the dense Jacobian.

        ``channels=None`` is the channel-free regression case; the others use a
        single spectrum shared by all channels.
        """
        spectrum = self._symmetric_spectrum(space_shape)
        channel_dim = 0 if channels is None else 1
        event_shape = space_shape if channels is None else space_shape + (channels,)

        bijection = SpectrumScaling(spectrum, channel_dim=channel_dim)
        x = jax.random.normal(rng_key, event_shape)

        _, ld = bijection.forward(x, jnp.zeros(()))
        np.testing.assert_allclose(
            -ld,
            self._dense_log_det(bijection, x),
            atol=ATOL_RELAXED,
            rtol=RTOL_RELAXED,
        )
        check_inverse(bijection, x)

    @pytest.mark.parametrize("space_shape", [(8,), (4, 4)])
    def test_spectrum_scaling_per_channel(self, rng_key, space_shape):
        """Per-channel spectra: one independent scaling per channel."""
        base = self._symmetric_spectrum(space_shape)
        # deliberately different scaling per channel
        spectrum = base[..., None] * jnp.array([0.4, 1.3, 2.7])

        bijection = SpectrumScaling(spectrum, channel_dim=1, space_dim=len(space_shape))
        x = jax.random.normal(rng_key, space_shape + (3,))

        _, ld = bijection.forward(x, jnp.zeros(()))
        np.testing.assert_allclose(
            -ld,
            self._dense_log_det(bijection, x),
            atol=ATOL_RELAXED,
            rtol=RTOL_RELAXED,
        )
        check_inverse(bijection, x)

    def test_spectrum_scaling_shared_spectrum_counts_channels(self, rng_key):
        """A shared spectrum contributes to log|det J| once *per channel*.

        Do not "simplify" this away: a fix that only repairs the broadcasting
        of the Fourier-space multiplication returns a log-det short by exactly
        the factor ``channels``, which this test is built to catch. The
        spectrum is chosen so that sum_k w_k log|s_k| is far from zero.
        """
        space_shape = (8,)
        channels = 3
        # scaled up so that sum_k w_k log|s_k| is comfortably away from zero
        spectrum = 1.7 * self._symmetric_spectrum(space_shape)

        x_single = jax.random.normal(rng_key, space_shape)
        x_multi = jnp.broadcast_to(x_single[..., None], space_shape + (channels,))

        single = SpectrumScaling(spectrum, channel_dim=0)
        shared = SpectrumScaling(spectrum, channel_dim=1)

        _, ld_single = single.forward(x_single, jnp.zeros(()))
        _, ld_shared = shared.forward(x_multi, jnp.zeros(()))

        # non-degenerate scaling: the missing factor would be visible
        assert abs(float(ld_single)) > 1.0
        np.testing.assert_allclose(
            ld_shared, channels * ld_single, atol=ATOL, rtol=RTOL
        )
        # and the same value from an independent Jacobian computation
        np.testing.assert_allclose(
            -ld_shared,
            self._dense_log_det(shared, x_multi),
            atol=ATOL_RELAXED,
            rtol=RTOL_RELAXED,
        )

    def test_spectrum_scaling_rejects_bad_rank(self):
        """Scaling rank matching neither shared nor per-channel layout fails."""
        spectrum = self._symmetric_spectrum((8,))
        # rank 1 declared as 2 space dims: neither 2 nor 2 + 1
        bijection = SpectrumScaling(spectrum, channel_dim=1, space_dim=2)
        with pytest.raises(ValueError, match="matches neither"):
            bijection.forward(jnp.ones((4, 4, 2)), jnp.zeros(()))

    def test_free_theory_scaling_channels(self, rng_key):
        """FreeTheoryScaling shares its spectrum across channels."""
        space_shape = (4, 4)
        bijection = FreeTheoryScaling(0.7, space_shape, channel_dim=1)
        x = jax.random.normal(rng_key, space_shape + (2,))

        _, ld = bijection.forward(x, jnp.zeros(()))
        np.testing.assert_allclose(
            -ld,
            self._dense_log_det(bijection, x),
            atol=ATOL_RELAXED,
            rtol=RTOL_RELAXED,
        )
        check_inverse(bijection, x)

    @pytest.mark.parametrize("space_shape", [(4, 4), (4, 4, 4)])
    @pytest.mark.parametrize(
        "case", ["pair_consistent", "perturb_paired", "perturb_unpaired"]
    )
    def test_spectrum_scaling_pair_precondition(self, rng_key, space_shape, case):
        """Pins the *exact* precondition on the spectrum.

        Only rFFT entries whose conjugate partner is also stored in the grid
        (``FourierMeta.copy_from``/``copy_to``) are constrained. An otherwise
        wildly k-asymmetric spectrum is perfectly fine, and perturbing an
        unpaired entry stays exact; only a perturbation of a paired entry
        breaks invertibility and the log-det. Do not "strengthen" this into
        a requirement that the spectrum be symmetric under k -> -k: that would
        reject legitimate spectra.
        """
        meta = FourierMeta.create(space_shape)
        assert len(meta.copy_to) > 0, "shape must store conjugate pairs"
        copy_to = tuple(meta.copy_to.T)
        copy_from = tuple(meta.copy_from.T)

        key_s, key_x = jax.random.split(rng_key)
        # deliberately asymmetric in k, but consistent on the stored pairs
        raw = jnp.exp(0.3 * jax.random.normal(key_s, meta.mr.shape))
        spectrum = raw.at[copy_to].set(raw[copy_from])

        if case == "perturb_paired":
            spectrum = spectrum.at[tuple(i[:1] for i in copy_to)].multiply(1.5)
        elif case == "perturb_unpaired":
            paired = {tuple(i) for i in np.concatenate([meta.copy_to, meta.copy_from])}
            idx = next(i for i in np.ndindex(meta.mr.shape) if i not in paired)
            spectrum = spectrum.at[idx].multiply(1.5)

        bijection = SpectrumScaling(spectrum)
        x = jax.random.normal(key_x, space_shape)
        _, ld = bijection.forward(x, jnp.zeros(()))
        dense = self._dense_log_det(bijection, x)
        asymmetry = spectrum_asymmetry(spectrum, space_shape)

        if case == "perturb_paired":
            assert float(asymmetry) > 0.1
            # silent failure without the precondition: log-det and inverse
            assert abs(float(-ld) - float(dense)) > 1e-3
            with pytest.raises(AssertionError):
                check_inverse(bijection, x)
        else:
            np.testing.assert_allclose(asymmetry, 0.0, atol=ATOL)
            np.testing.assert_allclose(-ld, dense, atol=ATOL_RELAXED, rtol=RTOL_RELAXED)
            check_inverse(bijection, x)

    def test_spectrum_asymmetry_vacuous_in_1d(self, rng_key):
        """No conjugate pairs are stored in one space dimension."""
        space_shape = (8,)
        spectrum = jnp.exp(0.3 * jax.random.normal(rng_key, (space_shape[0] // 2 + 1,)))
        assert float(spectrum_asymmetry(spectrum, space_shape)) == 0.0
        # ... while a hand-broken 2D pair does violate it. Momenta from
        # ``fft_momenta`` never do, at either ``lattice`` setting, since they
        # are folded into the first Brillouin zone.
        k = fft_momenta((4, 4))
        good = jnp.sum(k**2, axis=-1)
        assert float(spectrum_asymmetry(good, (4, 4))) == 0.0
        assert float(spectrum_asymmetry(good.at[1, 0].multiply(1.5), (4, 4))) > 0.1

    @pytest.mark.parametrize("space_shape", [(8,), (4, 4), (6, 6, 6)])
    def test_default_momenta_give_a_valid_spectrum(self, rng_key, space_shape):
        """A spectrum built from plain ``fft_momenta`` is a valid bijection.

        This is the regression for the momentum folding: with unfolded momenta
        the spectrum differs across conjugate pairs in two or more space
        dimensions, and both the log-density and invertibility break silently
        (measured before the fix, on 4x4: log-det off by 2.25 nats, round trip
        by 0.76).
        """
        k = fft_momenta(space_shape)
        spectrum = jnp.exp(-0.1 * jnp.sum(k**2, axis=-1)) + 0.2
        assert float(spectrum_asymmetry(spectrum, space_shape)) == 0.0

        bijection = SpectrumScaling(spectrum, space_dim=len(space_shape))
        x = jax.random.normal(rng_key, space_shape)
        _, ld = bijection.forward(x, jnp.zeros(()))
        np.testing.assert_allclose(
            -ld,
            self._dense_log_det(bijection, x),
            atol=ATOL_RELAXED,
            rtol=RTOL_RELAXED,
        )
        check_inverse(bijection, x)

    def test_per_k_class_spectrum_is_valid(self, rng_key):
        """A spectrum parametrised per |k| class is valid by construction.

        This is the pattern a *learnable* isotropic spectrum needs: gradients
        would otherwise drive conjugate partners apart. It works because
        ``FourierMeta.ks_full`` folds its indices, so partners share a class.
        """
        space_shape = (8, 8)
        meta = FourierMeta.create(space_shape)
        mr = np.asarray(meta.mr)
        cf, ct = np.asarray(meta.copy_from), np.asarray(meta.copy_to)

        classes = np.zeros(mr.shape, dtype=int)
        classes[mr] = np.asarray(meta.unique_unfold)
        classes[tuple(ct.T)] = classes[tuple(cf.T)]
        coefficients = np.exp(0.3 * np.sin(np.arange(classes.max() + 1)))
        spectrum = jnp.asarray(coefficients[classes])

        assert float(spectrum_asymmetry(spectrum, space_shape)) == 0.0
        bijection = SpectrumScaling(spectrum, space_dim=2)
        x = jax.random.normal(rng_key, space_shape)
        _, ld = bijection.forward(x, jnp.zeros(()))
        np.testing.assert_allclose(
            -ld,
            self._dense_log_det(bijection, x),
            atol=ATOL_RELAXED,
            rtol=RTOL_RELAXED,
        )
        check_inverse(bijection, x)

    @pytest.mark.parametrize("space_shape", [(8,), (4, 4)])
    def test_complex_spectrum_must_be_real_at_self_conjugate_modes(
        self, rng_key, space_shape
    ):
        """The pair condition alone does not certify a *complex* spectrum.

        Self-conjugate modes (``mr & ~mi``, e.g. k=0 and k=L/2) carry no
        imaginary degree of freedom, so a phase there breaks invertibility and
        the log-density even though every stored conjugate pair agrees. This
        binds in one space dimension too, where no pair is stored at all --
        which is why ``spectrum_asymmetry`` measures both conditions and why a
        version of it that only compared the pairs would return 0.0 for a
        spectrum that is not usable. Do not "simplify" it back.
        """
        meta = FourierMeta.create(space_shape)
        mr, mi = np.asarray(meta.mr), np.asarray(meta.mi)
        self_conj = mr & ~mi
        copy_from, copy_to = np.asarray(meta.copy_from), np.asarray(meta.copy_to)

        spectrum = np.asarray(
            jnp.exp(0.2 * jax.random.normal(rng_key, mr.shape))
        ).astype(complex)
        spectrum *= np.exp(1j * 0.4)  # phase everywhere, including self-conjugate
        if len(copy_to):  # make the pair condition hold exactly
            spectrum[tuple(copy_to.T)] = np.conj(spectrum[tuple(copy_from.T)])

        bad = jnp.asarray(spectrum)
        assert float(spectrum_asymmetry(bad, space_shape)) > 0.1
        with pytest.raises(AssertionError):
            check_inverse(
                SpectrumScaling(bad, space_dim=len(space_shape)),
                jax.random.normal(rng_key, space_shape),
            )

        good = jnp.asarray(np.where(self_conj, np.abs(spectrum), spectrum))
        assert float(spectrum_asymmetry(good, space_shape)) == 0.0
        bij = SpectrumScaling(good, space_dim=len(space_shape))
        x = jax.random.normal(rng_key, space_shape)
        _, log_det = bij.forward(x, jnp.zeros(()))
        np.testing.assert_allclose(
            -log_det, self._dense_log_det(bij, x), atol=ATOL, rtol=RTOL
        )
        check_inverse(bij, x)

    @pytest.mark.parametrize("m2", [0.7, nnx.Param(jnp.array(0.7))])
    def test_free_theory_scaling_lazy_spectrum(self, rng_key, m2):
        """Spectrum computed on the fly agrees with the precomputed one."""
        space_shape = (4, 4)
        x = jax.random.normal(rng_key, space_shape)

        eager = FreeTheoryScaling(0.7, space_shape)
        lazy = FreeTheoryScaling(m2, space_shape, precompute_spectrum=False)

        y_eager, ld_eager = eager.forward(x, jnp.zeros(()))
        y_lazy, ld_lazy = lazy.forward(x, jnp.zeros(()))

        np.testing.assert_allclose(y_lazy, y_eager, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld_lazy, ld_eager, atol=ATOL, rtol=RTOL)


class TestAdvancedCoupling:
    """Tests for advanced coupling layer functionality."""

    def test_module_reconstructor_basic(self, rng_key):
        """Test ModuleReconstructor parameter extraction."""
        # Create a bijection to extract parameters from
        bijection = AffineLinear(rngs=nnx.Rngs(rng_key))
        template = ModuleReconstructor(bijection)

        # Check parameter extraction
        assert template.params_total_size > 0
        assert len(template.params_shapes) > 0
        assert isinstance(template.params_dict, dict)
        assert not template.has_complex_params

        # Test reconstruction from array
        param_array = jnp.zeros(template.params_total_size)
        reconstructed = template.from_params(param_array)

        # Should be able to use the reconstructed bijection
        x = jnp.array([1.0])
        y, ld = reconstructed.forward(x, jnp.array(0.0))
        assert_finite_and_real(y, "reconstructed bijection output")
        assert_finite_and_real(ld, "reconstructed bijection log density")

    def test_module_reconstructor_with_spline(self, rng_key):
        """Test ModuleReconstructor with more complex bijection."""
        # Create spline bijection
        spline = MonotoneRQSpline(5, (), rngs=nnx.Rngs(rng_key))
        template = ModuleReconstructor(spline)

        # Test different parameter representations
        rng_key, k_arr = jax.random.split(rng_key)
        param_array = jax.random.normal(k_arr, (template.params_total_size,))

        # Test array reconstruction
        reconstructed_from_array = template.from_params(param_array)

        # Test dict reconstruction
        keys = jax.random.split(rng_key, len(template.params_shape_dict))
        param_dict = {
            key: jax.random.normal(k, shape)
            for (key, shape), k in zip(template.params_shape_dict.items(), keys)
        }
        reconstructed_from_dict = template.from_params(param_dict)

        # Test list reconstruction
        keys = jax.random.split(rng_key, len(template.params_shapes))
        param_list = [
            jax.random.normal(k, shape)
            for shape, k in zip(template.params_shapes, keys)
        ]
        reconstructed_from_list = template.from_params(param_list)

        # All should work for basic forward pass
        x = jnp.array([0.5])
        log_density = jnp.array(0.0)

        for reconstructed in [
            reconstructed_from_array,
            reconstructed_from_dict,
            reconstructed_from_list,
        ]:
            y, ld = reconstructed.forward(x, log_density)
            assert_finite_and_real(y, "reconstructed forward output")
            assert_finite_and_real(ld, "reconstructed forward log density")


class TestAdvancedIntegration:
    """Integration tests for advanced bijection combinations."""

    def test_fourier_spline_chain(self, rng_key):
        """Test chaining Fourier and spline bijections."""
        field_shape = (4, 4)
        rngs = nnx.Rngs(rng_key)

        spline = MonotoneRQSpline(8, field_shape, rngs=rngs)

        # Create chain: Fourier scaling -> flatten -> spline
        k_grid = fft_momenta(field_shape)
        k_squared = jnp.sum(k_grid**2, axis=-1)
        scaling = jnp.exp(-0.05 * k_squared)

        # Note: This is a conceptual test - actual chaining of these specific
        # bijections might require shape adjustments in practice
        spectrum_bij = SpectrumScaling(scaling, channel_dim=0)

        chain = Chain(
            spline,
            spectrum_bij,
        )

        x = jax.random.normal(rngs(), field_shape)
        log_density = jnp.array(0.0)

        # Test basic forward pass works
        y, ld_forward = chain.forward(x, log_density)
        assert_finite_and_real(y, "fourier chain forward output")
        assert_finite_and_real(ld_forward, "fourier chain forward log density")

    def test_advanced_bijection_gradient_flow(self):
        """Test gradient flow through advanced bijections."""

        def loss_fn(mass_param):
            # Create bijection with trainable parameter
            bijection = FreeTheoryScaling(mass_param, (4, 4), channel_dim=0)

            # Simple loss: transform a field and compute squared norm
            field = jnp.ones((4, 4))
            y, ld = bijection.forward(field, jnp.array(0.0))
            return jnp.sum(y**2) - ld

        mass_param = jnp.array(0.5)

        # Compute gradient
        loss_val, grad = jax.value_and_grad(loss_fn)(mass_param)

        assert_finite_and_real(jnp.array(loss_val), "advanced bijection loss")
        assert_finite_and_real(grad, "advanced bijection gradient")
        assert grad.shape == mass_param.shape

    def test_continuous_flow_batching(self):
        """Test continuous flow with batched inputs."""

        class SimpleBatchVF(nnx.Module):
            """Vector field that handles batched inputs."""

            def __call__(self, t, x, **kwargs):
                # Simple linear decay: dx/dt = -0.3 * x
                dx_dt = -0.3 * x
                # For batched inputs, return appropriate log density shape
                batch_shape = x.shape[:-1] if x.ndim > 0 else ()
                dlogp_dt = jnp.full(batch_shape, 0.3)
                return dx_dt, dlogp_dt

        vf = SimpleBatchVF()
        config = DiffraxConfig(
            t_start=0.0,
            t_end=1.0,
            dt=0.2,
            solver=diffrax.Euler(),
            stepsize_controller=diffrax.ConstantStepSize(),
        )
        flow = ContFlowDiffrax(vf, config)

        # Test with batched inputs
        batch_size = 3
        x = jnp.ones((batch_size, 2))  # batch of 2D vectors
        log_density = jnp.zeros(batch_size)

        # Test forward
        y, ld_forward = flow.forward(x, log_density)
        assert y.shape == x.shape
        assert ld_forward.shape == log_density.shape
        assert_finite_and_real(y, "batched flow forward output")
        assert_finite_and_real(ld_forward, "batched flow forward log density")
