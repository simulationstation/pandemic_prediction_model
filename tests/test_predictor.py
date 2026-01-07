"""
Tests for the pandemic predictor model.

Tests cover:
1. Bug regression (malicious_fraction must affect output)
2. Monotonicity expectations (parameter directions)
3. Boundary conditions
4. Hazard-to-probability conversion
5. Helper function correctness
"""

import pytest
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from predictor.predictor import (
    calculate_pandemic_probability,
    compute_iq_proportions,
    compute_capable_pool,
    compute_natural_hazard,
    compute_accidental_hazard,
    compute_malicious_hazard,
    hazard_to_probability,
    compute_cumulative_probability,
    run_sensitivity_analysis,
    run_detailed_analysis,
)


# =============================================================================
# BUG REGRESSION TESTS
# =============================================================================

class TestMaliciousFractionBug:
    """Test that malicious_fraction parameter actually affects output (bug regression)."""

    def test_malicious_fraction_affects_output(self):
        """Changing malicious_fraction should change the cumulative probability."""
        # Run with default malicious_fraction
        cum_prob_default, _ = calculate_pandemic_probability(malicious_fraction=0.00001)

        # Run with higher malicious_fraction
        cum_prob_higher, _ = calculate_pandemic_probability(malicious_fraction=0.001)

        # Run with lower malicious_fraction
        cum_prob_lower, _ = calculate_pandemic_probability(malicious_fraction=0.000001)

        # All three should be different
        assert cum_prob_higher != cum_prob_default, \
            "Higher malicious_fraction should produce different probability"
        assert cum_prob_lower != cum_prob_default, \
            "Lower malicious_fraction should produce different probability"

    def test_malicious_fraction_direction(self):
        """Higher malicious_fraction should increase risk (more risk from malicious pathway)."""
        cum_prob_low, _ = calculate_pandemic_probability(malicious_fraction=0.0)
        cum_prob_high, _ = calculate_pandemic_probability(malicious_fraction=0.5)

        assert cum_prob_high > cum_prob_low, \
            "Higher malicious_fraction should increase cumulative probability"


# =============================================================================
# MONOTONICITY TESTS
# =============================================================================

class TestMonotonicity:
    """Test that parameters affect output in expected directions."""

    def test_lab_growth_increases_engineered_risk(self):
        """Increasing lab_growth should not decrease engineered risk."""
        cum_prob_low, _ = calculate_pandemic_probability(lab_growth=0.05)
        cum_prob_high, _ = calculate_pandemic_probability(lab_growth=0.15)

        assert cum_prob_high >= cum_prob_low, \
            "Higher lab_growth should not decrease cumulative probability"

    def test_mitigation_growth_decreases_accidental_risk(self):
        """Increasing mitigation_growth should not increase accidental risk."""
        # Use high accidental fraction to isolate the effect
        cum_prob_low_mit, _ = calculate_pandemic_probability(
            mitigation_growth=0.01,
            malicious_fraction=0.0,  # No malicious, only accidental
            natural_prob_fraction=0.0  # No natural, only engineered
        )
        cum_prob_high_mit, _ = calculate_pandemic_probability(
            mitigation_growth=0.20,
            malicious_fraction=0.0,
            natural_prob_fraction=0.0
        )

        assert cum_prob_high_mit <= cum_prob_low_mit, \
            "Higher mitigation_growth should not increase accidental risk"

    def test_ai_growth_increases_capability_risk(self):
        """Increasing ai_growth should not decrease engineered risk."""
        cum_prob_low, _ = calculate_pandemic_probability(ai_growth=0.05)
        cum_prob_high, _ = calculate_pandemic_probability(ai_growth=0.25)

        assert cum_prob_high >= cum_prob_low, \
            "Higher ai_growth should not decrease cumulative probability"

    def test_base_annual_prob_increases_total_risk(self):
        """Increasing base_annual_prob should increase total risk."""
        cum_prob_low, _ = calculate_pandemic_probability(base_annual_prob=0.01)
        cum_prob_high, _ = calculate_pandemic_probability(base_annual_prob=0.05)

        assert cum_prob_high > cum_prob_low, \
            "Higher base_annual_prob should increase cumulative probability"

    def test_period_years_increases_cumulative_risk(self):
        """Longer time periods should have higher cumulative probability."""
        cum_prob_short, _ = calculate_pandemic_probability(period_years=5)
        cum_prob_long, _ = calculate_pandemic_probability(period_years=20)

        assert cum_prob_long > cum_prob_short, \
            "Longer period should have higher cumulative probability"

    def test_malicious_growth_increases_risk(self):
        """Increasing malicious_growth should increase risk (when malicious_fraction > 0)."""
        cum_prob_low, _ = calculate_pandemic_probability(
            malicious_fraction=0.1,  # Significant malicious component
            malicious_growth=0.01
        )
        cum_prob_high, _ = calculate_pandemic_probability(
            malicious_fraction=0.1,
            malicious_growth=0.10
        )

        assert cum_prob_high >= cum_prob_low, \
            "Higher malicious_growth should not decrease risk"


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestHazardToProbability:
    """Test the hazard to probability conversion."""

    def test_zero_hazard_gives_zero_probability(self):
        """Zero hazard rate should give zero probability."""
        assert hazard_to_probability(0.0) == 0.0

    def test_small_hazard_approximates_linearly(self):
        """For small hazard rates, P ≈ λ (linear approximation)."""
        small_hazard = 0.01
        prob = hazard_to_probability(small_hazard)
        # For small λ, 1 - exp(-λ) ≈ λ
        assert abs(prob - small_hazard) < 0.001

    def test_large_hazard_approaches_one(self):
        """Large hazard rates should give probability approaching 1."""
        large_hazard = 10.0
        prob = hazard_to_probability(large_hazard)
        assert prob > 0.999

    def test_negative_hazard_raises_error(self):
        """Negative hazard rate should raise ValueError."""
        with pytest.raises(ValueError):
            hazard_to_probability(-0.1)

    def test_known_values(self):
        """Test against known mathematical values."""
        # P = 1 - exp(-1) ≈ 0.6321
        assert abs(hazard_to_probability(1.0) - (1 - math.exp(-1))) < 1e-10


class TestCumulativeProbability:
    """Test cumulative probability calculation."""

    def test_single_year(self):
        """Single year should return that year's probability."""
        prob = compute_cumulative_probability([0.1])
        assert abs(prob - 0.1) < 1e-10

    def test_two_independent_years(self):
        """Two years: P(at least one) = 1 - (1-p1)(1-p2)."""
        probs = [0.1, 0.2]
        expected = 1 - (0.9 * 0.8)  # = 0.28
        actual = compute_cumulative_probability(probs)
        assert abs(actual - expected) < 1e-10

    def test_all_zeros(self):
        """All zero probabilities should give zero cumulative."""
        prob = compute_cumulative_probability([0.0, 0.0, 0.0])
        assert prob == 0.0

    def test_certainty_in_one_year(self):
        """If any year has P=1, cumulative should be 1."""
        prob = compute_cumulative_probability([0.1, 1.0, 0.2])
        assert prob == 1.0


class TestIQProportions:
    """Test IQ proportion calculations."""

    def test_standard_normal_proportions(self):
        """Test with standard IQ distribution."""
        prop_high, prop_mid, ratio = compute_iq_proportions(
            iq_mean=100, iq_sd=15, iq_high=130, iq_mid=110
        )

        # IQ 130 is 2 SD above mean, ~2.28% of population
        assert 0.02 < prop_high < 0.03

        # IQ 110-130 is between 0.67 SD and 2 SD, ~22% of population
        assert 0.20 < prop_mid < 0.25

        # Ratio should be roughly 10x (22% / 2.3%)
        assert 8 < ratio < 12

    def test_ratio_represents_ai_expansion(self):
        """The ratio should represent how much AI expands capable pool."""
        _, _, ratio = compute_iq_proportions(
            iq_mean=100, iq_sd=15, iq_high=130, iq_mid=110
        )

        # At full AI adoption, capable pool expands by (1 + ratio)
        # This should be roughly 10x expansion
        assert ratio > 5  # At least 5x expansion potential


class TestCapablePool:
    """Test capable pool calculations."""

    def test_baseline_returns_base_capable(self):
        """At baseline conditions (t=0), should return approximately base_capable."""
        _, _, ratio = compute_iq_proportions(100, 15, 130, 110)

        capable = compute_capable_pool(
            pop=8.3e9,
            prop_high=0.0228,  # ~2.28% at IQ 130
            ai_capability_ratio=ratio,
            ai_adoption=0.5,
            base_capable=30000,
            base_pop=8.3e9
        )

        # At 50% AI adoption with ratio ~10, expansion is ~6x
        # So capable pool is ~30000 * 6 = 180000
        assert capable > 30000  # At least base_capable
        assert capable < 500000  # But not unreasonably high

    def test_population_growth_scales_capable(self):
        """Population growth should proportionally scale capable pool."""
        _, _, ratio = compute_iq_proportions(100, 15, 130, 110)

        capable_base = compute_capable_pool(
            pop=8.3e9, prop_high=0.0228, ai_capability_ratio=ratio,
            ai_adoption=0.5, base_capable=30000, base_pop=8.3e9
        )

        capable_doubled = compute_capable_pool(
            pop=16.6e9, prop_high=0.0228, ai_capability_ratio=ratio,
            ai_adoption=0.5, base_capable=30000, base_pop=8.3e9
        )

        assert abs(capable_doubled / capable_base - 2.0) < 0.01


class TestHazardComponents:
    """Test individual hazard component functions."""

    def test_natural_hazard_is_constant(self):
        """Natural hazard should just return the base rate."""
        assert compute_natural_hazard(0.015) == 0.015
        assert compute_natural_hazard(0.0) == 0.0

    def test_accidental_hazard_scales_correctly(self):
        """Accidental hazard should scale with labs/capability, inverse with mitigation."""
        base = 0.01

        # Double labs = double hazard
        h1 = compute_accidental_hazard(base, lab_multiplier=1.0, capability_multiplier=1.0, mitigation_factor=1.0)
        h2 = compute_accidental_hazard(base, lab_multiplier=2.0, capability_multiplier=1.0, mitigation_factor=1.0)
        assert abs(h2 / h1 - 2.0) < 0.01

        # Double mitigation = half hazard
        h3 = compute_accidental_hazard(base, lab_multiplier=1.0, capability_multiplier=1.0, mitigation_factor=2.0)
        assert abs(h3 / h1 - 0.5) < 0.01

    def test_malicious_hazard_scales_correctly(self):
        """Malicious hazard should scale with labs, capability, and incentives."""
        base = 0.001

        # Double incentives = double hazard
        h1 = compute_malicious_hazard(base, lab_multiplier=1.0, capability_multiplier=1.0, malicious_incentive_factor=1.0)
        h2 = compute_malicious_hazard(base, lab_multiplier=1.0, capability_multiplier=1.0, malicious_incentive_factor=2.0)
        assert abs(h2 / h1 - 2.0) < 0.01


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the full model."""

    def test_default_params_produce_reasonable_output(self):
        """Default parameters should produce reasonable probability."""
        cum_prob, annual_probs = calculate_pandemic_probability()

        # Should be between 0 and 100%
        assert 0 < cum_prob < 100

        # Should have correct number of annual probs
        assert len(annual_probs) == 10

        # Annual probs should all be valid probabilities
        for p in annual_probs:
            assert 0 <= p <= 1

    def test_annual_probs_sum_to_cumulative(self):
        """Cumulative probability should match annual probs calculation."""
        cum_prob, annual_probs = calculate_pandemic_probability()

        # Recalculate cumulative from annual
        expected_cum = compute_cumulative_probability(annual_probs) * 100

        assert abs(cum_prob - expected_cum) < 0.01

    def test_detailed_analysis_matches_standard(self):
        """Detailed analysis should match standard calculation."""
        cum_prob, _ = calculate_pandemic_probability()
        detailed = run_detailed_analysis()

        assert abs(cum_prob - detailed['cumulative_prob_pct']) < 0.01

    def test_zero_engineered_risk(self):
        """With natural_prob_fraction=1, only natural risk contributes."""
        cum_prob, annual_probs = calculate_pandemic_probability(
            natural_prob_fraction=1.0,
            base_annual_prob=0.02
        )

        # Natural hazard = 0.02, P = 1 - exp(-0.02) ≈ 0.0198
        expected_annual = 1 - math.exp(-0.02)
        assert abs(annual_probs[0] - expected_annual) < 0.001

    def test_zero_natural_risk(self):
        """With natural_prob_fraction=0, only engineered risk contributes."""
        cum_prob, _ = calculate_pandemic_probability(
            natural_prob_fraction=0.0,
            base_annual_prob=0.02
        )

        # Should still produce valid output
        assert 0 <= cum_prob <= 100


# =============================================================================
# SENSITIVITY ANALYSIS TESTS
# =============================================================================

class TestSensitivityAnalysis:
    """Test Monte Carlo sensitivity analysis."""

    def test_sensitivity_returns_valid_structure(self):
        """Sensitivity analysis should return expected structure."""
        results = run_sensitivity_analysis(n_samples=10, seed=42)

        assert 'mean' in results
        assert 'std' in results
        assert 'percentiles' in results
        assert 'cumulative_probs' in results

        assert len(results['cumulative_probs']) == 10

    def test_sensitivity_with_custom_distributions(self):
        """Should work with custom parameter distributions."""
        import numpy as np

        distributions = {
            'base_annual_prob': lambda: np.random.uniform(0.02, 0.03)
        }

        results = run_sensitivity_analysis(
            n_samples=10,
            param_distributions=distributions,
            seed=42
        )

        assert results['mean'] > 0

    def test_sensitivity_reproducibility(self):
        """Same seed should produce same results."""
        results1 = run_sensitivity_analysis(n_samples=10, seed=123)
        results2 = run_sensitivity_analysis(n_samples=10, seed=123)

        assert results1['mean'] == results2['mean']


# =============================================================================
# BOUNDARY TESTS
# =============================================================================

class TestBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_single_year(self):
        """Should work with period_years=1."""
        cum_prob, annual_probs = calculate_pandemic_probability(period_years=1)
        assert len(annual_probs) == 1
        assert 0 <= cum_prob <= 100

    def test_long_period(self):
        """Should work with long time periods."""
        cum_prob, annual_probs = calculate_pandemic_probability(period_years=100)
        assert len(annual_probs) == 100
        assert cum_prob <= 100  # Cannot exceed 100%

    def test_high_base_prob(self):
        """Should handle high base probabilities gracefully."""
        cum_prob, annual_probs = calculate_pandemic_probability(base_annual_prob=0.5)
        for p in annual_probs:
            assert p <= 1.0  # Probability capped at 1

    def test_zero_growth_rates(self):
        """Should work with zero growth rates (static model)."""
        cum_prob, annual_probs = calculate_pandemic_probability(
            pop_growth=0.0,
            ai_growth=0.0,
            lab_growth=0.0,
            mitigation_growth=0.0,
            malicious_growth=0.0
        )

        # All annual probs should be identical (static)
        assert all(abs(p - annual_probs[0]) < 1e-10 for p in annual_probs)


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])
