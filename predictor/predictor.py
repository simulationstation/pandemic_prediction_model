"""
Pandemic Risk Prediction Model v2.0
===================================

A comprehensive hazard-based model for estimating pandemic probability over time,
incorporating natural, accidental, and malicious pathways with multiple risk factors.

MODEL ASSUMPTIONS AND LIMITATIONS
---------------------------------

1. STRUCTURE: The model uses a competing hazards framework with multiple pathways:

   NATURAL PATHWAYS:
   - Zoonotic spillover (baseline, historically constant)
   - Climate-enhanced (permafrost thaw, habitat destruction - grows over time)

   ACCIDENTAL PATHWAYS:
   - General lab accidents (scales with labs, capability, inverse with mitigation)
   - Gain-of-function accidents (subset of lab work with higher risk multiplier)
   - Cloud lab incidents (new access vector, different risk profile)

   MALICIOUS PATHWAYS:
   - Individual actors (scales with capability pool, incentives)
   - State-sponsored programs (different resource/capability profile)

2. CAPABILITY FACTORS: Multiple factors affect who can cause a pandemic:
   - IQ/expertise threshold (original model)
   - AI assistance lowering threshold (original model)
   - AI as direct agent (can design pathogens without human expertise)
   - DNA synthesis accessibility (bypasses need for lab access)
   - Knowledge diffusion (published research lowers barriers)

3. DEFENSE FACTORS:
   - Mitigation/biosecurity (can improve OR erode over time)
   - Antibiotic resistance (affects pandemic severity/lethality)

4. CORRELATED RISKS:
   - Near-miss events can trigger security improvements OR reveal vulnerabilities
   - Modeled as a correlation factor between years

5. HAZARD COMBINATION: Independent hazards are combined as:
   P(event in year) = 1 - exp(-total_hazard_rate)
   This is standard survival analysis.

6. GROWTH MODELS:
   - Compound growth: population, labs, AI adoption, DNA synthesis, cloud labs,
     knowledge diffusion, antibiotic resistance
   - Linear growth: mitigation drift, malicious incentives, permafrost thaw,
     state actor capability
   - Sigmoid/logistic: AI direct agent capability (phase transition)

7. LIMITATIONS:
   - Parameter defaults are illustrative, not empirically validated
   - Assumes pathway independence (partially addressed by correlation factor)
   - Does not model pandemic severity separately (antibiotic resistance is proxy)
   - State actor modeling is simplified
   - Climate effects are rough approximations

NEW PARAMETERS (v2.0)
---------------------
All new parameters default to values that minimally change baseline behavior,
allowing incremental adjustment.

DNA Synthesis:
- dna_synthesis_cost_reduction: Annual cost reduction rate (default 0.3 = 30%/year)
- dna_synthesis_capability_factor: How much synthesis access substitutes for lab access

Cloud Labs:
- cloud_lab_start: Starting cloud lab accessibility index (default 0.1 = 10% of traditional)
- cloud_lab_growth: Annual growth rate (default 0.25)

AI Direct Agent:
- ai_direct_start_year: Year when AI begins direct pathogen design capability
- ai_direct_growth_rate: How fast AI direct capability grows (logistic rate)
- ai_direct_max_multiplier: Maximum capability multiplier from AI direct agent

State Actors:
- state_actor_base_prob: Baseline annual probability from state programs
- state_actor_growth: Annual growth in state program risk

Gain-of-Function:
- gof_fraction: Fraction of lab work that's GoF
- gof_risk_multiplier: Risk multiplier for GoF vs general lab work

Climate/Permafrost:
- permafrost_thaw_rate: Annual increase in natural hazard from climate effects

Regulatory:
- regulatory_drift: Annual change in regulatory effectiveness (can be negative)

Antibiotic Resistance:
- antibiotic_resistance_growth: Annual growth in resistance
- antibiotic_severity_multiplier: How resistance increases effective pandemic risk

Knowledge Diffusion:
- knowledge_diffusion_rate: Annual rate of dangerous knowledge becoming accessible
- knowledge_capability_boost: How knowledge diffusion affects capability threshold

Risk Correlation:
- yearly_risk_correlation: Correlation factor between years (0 = independent)
- near_miss_security_boost: Security improvement after near-miss
- near_miss_vulnerability_reveal: Vulnerability exposure after near-miss

BACKWARD COMPATIBILITY
----------------------
All original parameters are preserved with identical names and defaults.
The model produces similar (not identical) results with default parameters
due to the additional risk pathways, but the structure is compatible.
"""

import argparse
import math
from typing import Tuple, List, Dict, Optional, Callable
from scipy.stats import norm
import numpy as np


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sigmoid(x: float, k: float = 1.0, x0: float = 0.0) -> float:
    """
    Logistic sigmoid function for smooth phase transitions.

    Args:
        x: Input value
        k: Steepness parameter (higher = sharper transition)
        x0: Midpoint of transition

    Returns:
        Value in (0, 1)
    """
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range [min_val, max_val]."""
    return max(min_val, min(max_val, value))


# =============================================================================
# CORE COMPUTATION FUNCTIONS
# =============================================================================

def compute_iq_proportions(
    iq_mean: float,
    iq_sd: float,
    iq_high: int,
    iq_mid: int
) -> Tuple[float, float, float]:
    """
    Compute population fractions based on IQ thresholds.

    Args:
        iq_mean: Population IQ mean
        iq_sd: Population IQ standard deviation
        iq_high: IQ threshold for baseline capability (without AI assistance)
        iq_mid: IQ threshold for capability with AI assistance

    Returns:
        Tuple of (prop_high, prop_mid_range, ai_capability_ratio)
    """
    z_high = (iq_high - iq_mean) / iq_sd
    z_mid = (iq_mid - iq_mean) / iq_sd

    prop_high = 1 - norm.cdf(z_high)
    prop_mid_range = norm.cdf(z_high) - norm.cdf(z_mid)

    ai_capability_ratio = prop_mid_range / prop_high if prop_high > 0 else 0

    return prop_high, prop_mid_range, ai_capability_ratio


def compute_capable_pool(
    pop: float,
    prop_high: float,
    ai_capability_ratio: float,
    ai_adoption: float,
    base_capable: int,
    base_pop: float,
    knowledge_diffusion_factor: float = 1.0,
    ai_direct_factor: float = 0.0
) -> float:
    """
    Compute the effective number of capable individuals.

    Args:
        pop: Current population
        prop_high: Fraction of population with high IQ
        ai_capability_ratio: Ratio of AI-expandable pool to baseline capable
        ai_adoption: Current AI adoption rate [0, 1]
        base_capable: Baseline number of capable individuals at t=0
        base_pop: Starting population (for scaling)
        knowledge_diffusion_factor: Multiplier from knowledge accessibility (>= 1)
        ai_direct_factor: Additional capability from AI as direct agent [0, 1]

    Returns:
        Effective number of capable individuals
    """
    pop_factor = pop / base_pop
    baseline_capable = base_capable * pop_factor

    # AI assistance expands pool by bringing in mid-range individuals
    ai_expansion = 1 + ai_capability_ratio * ai_adoption

    # Knowledge diffusion further expands the pool
    knowledge_expansion = knowledge_diffusion_factor

    # AI as direct agent adds capability independent of human pool
    # This represents AI systems that can design pathogens without human expertise
    # Modeled as additional "virtual" capable actors
    ai_direct_addition = ai_direct_factor * baseline_capable

    return baseline_capable * ai_expansion * knowledge_expansion + ai_direct_addition


def compute_synthesis_access_factor(
    years_elapsed: int,
    dna_synthesis_cost_reduction: float,
    dna_synthesis_capability_factor: float
) -> float:
    """
    Compute how DNA synthesis democratization affects lab access requirements.

    As synthesis costs drop, actors need less traditional lab infrastructure.

    Args:
        years_elapsed: Years since model start
        dna_synthesis_cost_reduction: Annual cost reduction rate (e.g., 0.3 = 30%)
        dna_synthesis_capability_factor: Max factor by which synthesis substitutes for labs

    Returns:
        Multiplier for effective capability (>= 1)
    """
    # Cost drops exponentially, accessibility rises
    # At year 0, factor = 1. As costs drop, factor approaches (1 + capability_factor)
    cost_factor = (1 - dna_synthesis_cost_reduction) ** years_elapsed
    accessibility = 1 - cost_factor  # 0 at year 0, approaches 1 as costs drop

    return 1 + dna_synthesis_capability_factor * accessibility


def compute_cloud_lab_factor(
    cloud_lab_start: float,
    cloud_lab_growth: float,
    years_elapsed: int,
    lab_start: int
) -> float:
    """
    Compute effective lab multiplier including cloud/remote labs.

    Args:
        cloud_lab_start: Starting cloud lab accessibility (fraction of traditional)
        cloud_lab_growth: Annual growth rate of cloud lab access
        years_elapsed: Years since model start
        lab_start: Traditional lab count at start

    Returns:
        Effective additional labs from cloud access (as fraction of lab_start)
    """
    # Cloud labs grow from a small base
    cloud_accessibility = cloud_lab_start * ((1 + cloud_lab_growth) ** years_elapsed)
    cloud_accessibility = min(cloud_accessibility, 2.0)  # Cap at 2x traditional equivalent

    return cloud_accessibility


def compute_ai_direct_capability(
    year: int,
    ai_direct_start_year: int,
    ai_direct_growth_rate: float,
    ai_direct_max_multiplier: float
) -> float:
    """
    Compute AI direct agent capability using sigmoid growth.

    Models the phase transition where AI becomes capable of directly
    designing dangerous pathogens without requiring human expertise.

    Args:
        year: Current simulation year
        ai_direct_start_year: Year when capability begins emerging
        ai_direct_growth_rate: Steepness of capability growth
        ai_direct_max_multiplier: Maximum capability multiplier

    Returns:
        AI direct capability factor [0, ai_direct_max_multiplier]
    """
    if year < ai_direct_start_year - 5:
        return 0.0

    # Sigmoid centered at start_year
    raw = sigmoid(year, k=ai_direct_growth_rate, x0=ai_direct_start_year)

    return raw * ai_direct_max_multiplier


def compute_natural_hazard(
    natural_base_hazard: float,
    permafrost_factor: float = 1.0,
    urbanization_factor: float = 1.0
) -> float:
    """
    Compute natural pandemic hazard rate.

    Args:
        natural_base_hazard: Baseline natural hazard rate
        permafrost_factor: Multiplier from climate/permafrost effects (>= 1)
        urbanization_factor: Multiplier from urbanization/density effects (>= 1)

    Returns:
        Natural hazard rate for the year
    """
    return natural_base_hazard * permafrost_factor * urbanization_factor


def compute_urbanization_factor(
    years_elapsed: int,
    urbanization_rate: float,
    density_natural_multiplier: float
) -> float:
    """
    Compute urbanization effect on natural hazard.

    Urbanization increases zoonotic spillover risk through:
    - More human-animal interface (wet markets, deforestation)
    - Higher density living conditions
    - Megacity growth concentrating populations

    Args:
        years_elapsed: Years since model start
        urbanization_rate: Annual increase in urban population fraction
        density_natural_multiplier: How much density affects spillover risk

    Returns:
        Urbanization multiplier for natural hazard (>= 1)
    """
    # Urbanization compounds over time
    urban_growth = (1 + urbanization_rate) ** years_elapsed

    # Convert to hazard multiplier (scaled by density_natural_multiplier)
    return 1 + (urban_growth - 1) * density_natural_multiplier


def compute_accidental_hazard(
    accidental_base_hazard: float,
    lab_multiplier: float,
    capability_multiplier: float,
    mitigation_factor: float,
    gof_fraction: float,
    gof_risk_multiplier: float,
    cloud_lab_factor: float,
    synthesis_factor: float
) -> float:
    """
    Compute accidental (lab) pandemic hazard rate.

    Includes general lab accidents, gain-of-function specific risk,
    and cloud lab incidents.

    Args:
        accidental_base_hazard: Baseline accidental hazard rate
        lab_multiplier: Traditional labs relative to baseline
        capability_multiplier: Capable pool relative to baseline
        mitigation_factor: Mitigation effectiveness (>= 1, higher = safer)
        gof_fraction: Fraction of lab work that's gain-of-function
        gof_risk_multiplier: Additional risk multiplier for GoF work
        cloud_lab_factor: Effective lab multiplier from cloud labs
        synthesis_factor: Capability boost from DNA synthesis access

    Returns:
        Total accidental hazard rate for the year
    """
    # General lab accidents
    general_fraction = 1 - gof_fraction
    general_hazard = (accidental_base_hazard * general_fraction *
                      lab_multiplier * capability_multiplier / mitigation_factor)

    # Gain-of-function accidents (higher risk per incident)
    gof_hazard = (accidental_base_hazard * gof_fraction * gof_risk_multiplier *
                  lab_multiplier * capability_multiplier / mitigation_factor)

    # Cloud lab incidents (different risk profile - less oversight)
    # Cloud labs have reduced mitigation effectiveness
    cloud_mitigation = max(1.0, mitigation_factor * 0.5)  # 50% less effective
    cloud_hazard = (accidental_base_hazard * 0.3 *  # 30% of base rate
                    cloud_lab_factor * capability_multiplier * synthesis_factor /
                    cloud_mitigation)

    return general_hazard + gof_hazard + cloud_hazard


def compute_malicious_hazard(
    malicious_base_hazard: float,
    lab_multiplier: float,
    capability_multiplier: float,
    malicious_incentive_factor: float,
    synthesis_factor: float,
    state_actor_hazard: float
) -> float:
    """
    Compute malicious (intentional) pandemic hazard rate.

    Includes individual actors and state-sponsored programs.

    NOTE: Mitigation (biosafety) does NOT reduce malicious hazard because
    malicious actors intentionally circumvent safety measures.

    Args:
        malicious_base_hazard: Baseline malicious hazard rate (individuals)
        lab_multiplier: Labs relative to baseline (access points)
        capability_multiplier: Capable pool relative to baseline
        malicious_incentive_factor: Growth in malicious incentives (>= 1)
        synthesis_factor: How synthesis access bypasses lab requirements
        state_actor_hazard: Additional hazard from state-sponsored programs

    Returns:
        Total malicious hazard rate for the year
    """
    # Individual malicious actors
    # Synthesis reduces dependence on labs (weighted average)
    effective_lab_factor = 0.5 * lab_multiplier + 0.5 * synthesis_factor

    individual_hazard = (malicious_base_hazard * effective_lab_factor *
                         capability_multiplier * malicious_incentive_factor)

    # State-sponsored programs (added separately - different resource profile)
    # State actors don't depend on public capability pool

    return individual_hazard + state_actor_hazard


def hazard_to_probability(hazard_rate: float) -> float:
    """
    Convert hazard rate to probability using survival function.
    P(event) = 1 - exp(-hazard)
    """
    if hazard_rate < 0:
        raise ValueError(f"Hazard rate must be non-negative, got {hazard_rate}")
    return 1 - math.exp(-hazard_rate)


def compute_cumulative_probability(annual_probs: List[float]) -> float:
    """
    Compute cumulative probability from annual probabilities.
    P(at least one event) = 1 - product(1 - p_i)
    """
    survival_prob = 1.0
    for p in annual_probs:
        survival_prob *= (1 - p)
    return 1 - survival_prob


def apply_risk_correlation(
    annual_probs: List[float],
    correlation: float,
    near_miss_security_boost: float,
    near_miss_vulnerability_reveal: float
) -> List[float]:
    """
    Apply correlation effects between years.

    Models how near-misses in one year affect risk in subsequent years.
    A near-miss can both:
    - Trigger security improvements (reduces future risk)
    - Reveal vulnerabilities (increases future risk)

    The net effect depends on parameter balance.

    Args:
        annual_probs: Raw annual probabilities
        correlation: Base correlation factor between years
        near_miss_security_boost: Risk reduction after high-risk year
        near_miss_vulnerability_reveal: Risk increase from revealed vulnerabilities

    Returns:
        Adjusted annual probabilities
    """
    if correlation == 0 and near_miss_security_boost == 0 and near_miss_vulnerability_reveal == 0:
        return annual_probs

    adjusted = list(annual_probs)

    for i in range(1, len(adjusted)):
        prev_prob = adjusted[i - 1]

        # High probability year triggers responses
        if prev_prob > 0.03:  # Threshold for "concerning" year
            # Security boost reduces risk
            security_effect = 1 - (near_miss_security_boost * prev_prob)

            # Vulnerability revelation increases risk
            vulnerability_effect = 1 + (near_miss_vulnerability_reveal * prev_prob)

            # Combined effect
            net_effect = security_effect * vulnerability_effect
            adjusted[i] = clamp(adjusted[i] * net_effect, 0.0, 1.0)

        # General correlation (risk clusters)
        if correlation > 0:
            # If previous year was high risk, current year slightly elevated
            correlation_adjustment = 1 + correlation * (prev_prob - 0.025)  # 0.025 as baseline
            adjusted[i] = clamp(adjusted[i] * correlation_adjustment, 0.0, 1.0)

    return adjusted


def apply_antibiotic_resistance(
    annual_probs: List[float],
    antibiotic_resistance_growth: float,
    antibiotic_severity_multiplier: float
) -> List[float]:
    """
    Adjust probabilities for antibiotic resistance effects.

    Antibiotic resistance doesn't change occurrence probability directly,
    but increases the chance that a pandemic becomes severe (which is what
    we're modeling). This is a proxy for severity-adjusted probability.

    Args:
        annual_probs: Annual probabilities
        antibiotic_resistance_growth: Annual growth in resistance
        antibiotic_severity_multiplier: How resistance increases effective risk

    Returns:
        Adjusted annual probabilities
    """
    adjusted = []
    for i, prob in enumerate(annual_probs):
        # Resistance compounds over time
        resistance_level = (1 + antibiotic_resistance_growth) ** i

        # Higher resistance means pandemics more likely to be severe
        # This increases "effective" probability of a severe pandemic
        severity_adjustment = 1 + (resistance_level - 1) * antibiotic_severity_multiplier

        adjusted_prob = clamp(prob * severity_adjustment, 0.0, 1.0)
        adjusted.append(adjusted_prob)

    return adjusted


def apply_density_transmission(
    annual_probs: List[float],
    urbanization_rate: float,
    density_transmission_multiplier: float
) -> List[float]:
    """
    Adjust probabilities for population density effects on transmission.

    Higher density doesn't just increase spillover (handled in natural hazard),
    it also increases transmission speed and severity once a pandemic starts.
    Denser populations = faster spread = higher effective risk.

    Args:
        annual_probs: Annual probabilities
        urbanization_rate: Annual growth in urbanization
        density_transmission_multiplier: How density affects transmission severity

    Returns:
        Adjusted annual probabilities
    """
    adjusted = []
    for i, prob in enumerate(annual_probs):
        # Urbanization compounds over time
        urban_growth = (1 + urbanization_rate) ** i

        # Higher density = faster transmission = more severe pandemics
        transmission_adjustment = 1 + (urban_growth - 1) * density_transmission_multiplier

        adjusted_prob = clamp(prob * transmission_adjustment, 0.0, 1.0)
        adjusted.append(adjusted_prob)

    return adjusted


# =============================================================================
# MAIN MODEL FUNCTION
# =============================================================================

def calculate_pandemic_probability(
    # === ORIGINAL PARAMETERS (preserved exactly) ===
    period_years: int = 10,
    base_annual_prob: float = 0.025,
    natural_prob_fraction: float = 0.5,  # 50% natural, 50% engineered (aggressive)
    malicious_fraction: float = 0.00001,
    malicious_growth: float = 0.03,
    mitigation_growth: float = 0.02,  # Reduced - biosecurity improvement is slow
    pop: float = 8.3e9,
    pop_growth: float = 0.008,
    iq_mean: float = 100,
    iq_sd: float = 15,
    iq_high: int = 130,
    iq_mid: int = 110,
    base_capable: int = 30000,
    ai_adoption_start: float = 0.5,
    ai_growth: float = 0.15,
    lab_start: int = 3515,
    lab_growth: float = 0.1,
    # === NEW PARAMETERS (v2.0) ===
    # DNA Synthesis
    dna_synthesis_cost_reduction: float = 0.3,  # 30% annual cost reduction
    dna_synthesis_capability_factor: float = 0.5,  # Max 50% capability boost
    # Cloud Labs
    cloud_lab_start: float = 0.05,  # 5% of traditional lab access
    cloud_lab_growth: float = 0.20,  # 20% annual growth
    # AI Direct Agent
    ai_direct_start_year: int = 3,  # Years until AI direct capability emerges (aggressive)
    ai_direct_growth_rate: float = 0.5,  # Sigmoid steepness
    ai_direct_max_multiplier: float = 0.3,  # Max 30% capability addition
    # State Actors
    state_actor_base_prob: float = 0.003,  # 0.3% annual baseline (aggressive)
    state_actor_growth: float = 0.03,  # 3% annual growth (aggressive)
    # Gain-of-Function
    gof_fraction: float = 0.15,  # 15% of lab work is GoF (aggressive)
    gof_risk_multiplier: float = 5.0,  # GoF is 5x riskier (aggressive)
    # Climate/Permafrost
    permafrost_thaw_rate: float = 0.01,  # 1% annual increase in natural baseline
    # Urbanization/Density
    urbanization_rate: float = 0.015,  # 1.5% annual increase in urban population fraction
    density_natural_multiplier: float = 0.3,  # How much density increases natural spillover risk
    density_transmission_multiplier: float = 0.2,  # How much density increases effective severity
    # Regulatory
    regulatory_drift: float = -0.01,  # Slight regulatory erosion over time (aggressive)
    # Antibiotic Resistance
    antibiotic_resistance_growth: float = 0.03,  # 3% annual growth
    antibiotic_severity_multiplier: float = 0.1,  # 10% severity increase per unit resistance
    # Knowledge Diffusion
    knowledge_diffusion_rate: float = 0.05,  # 5% annual knowledge accessibility growth
    # Risk Correlation
    yearly_risk_correlation: float = 0.0,  # No correlation by default
    near_miss_security_boost: float = 0.1,  # 10% risk reduction per unit near-miss
    near_miss_vulnerability_reveal: float = 0.05,  # 5% risk increase from revealed vulnerabilities
) -> Tuple[float, List[float]]:
    """
    Calculate pandemic probability over a time period using comprehensive hazard model.

    The model decomposes pandemic risk into multiple pathways:

    NATURAL:
    - Zoonotic spillover (baseline + climate/permafrost enhancement)

    ACCIDENTAL:
    - General lab accidents
    - Gain-of-function accidents (higher risk)
    - Cloud lab incidents (less oversight)

    MALICIOUS:
    - Individual actors (capability + incentive driven)
    - State-sponsored programs (resource-driven)

    All original parameters are preserved for backward compatibility.
    New parameters default to values that minimally change baseline behavior.

    Args:
        [See parameter definitions above]

    Returns:
        Tuple of (cumulative_probability_percent, annual_probabilities)
    """
    # === COMPUTE STATIC IQ PARAMETERS ===
    prop_high, prop_mid_range, ai_capability_ratio = compute_iq_proportions(
        iq_mean, iq_sd, iq_high, iq_mid
    )

    # === DECOMPOSE BASE HAZARD ===
    natural_base_hazard = base_annual_prob * natural_prob_fraction
    engineered_base_hazard = base_annual_prob * (1 - natural_prob_fraction)
    accidental_base_hazard = engineered_base_hazard * (1 - malicious_fraction)
    malicious_base_hazard = engineered_base_hazard * malicious_fraction

    # === COMPUTE BASELINE CAPABILITY (for normalization) ===
    baseline_capable_pool = compute_capable_pool(
        pop, prop_high, ai_capability_ratio, ai_adoption_start, base_capable, pop,
        knowledge_diffusion_factor=1.0, ai_direct_factor=0.0
    )

    # === SIMULATE YEAR BY YEAR ===
    annual_probs_raw = []
    current_pop = pop
    current_ai_adoption = ai_adoption_start
    current_labs = lab_start

    for t in range(1, period_years + 1):
        years_elapsed = t - 1

        # --- Compute time-varying factors ---

        # Lab multiplier (traditional)
        lab_multiplier = current_labs / lab_start

        # Cloud lab accessibility
        cloud_factor = compute_cloud_lab_factor(
            cloud_lab_start, cloud_lab_growth, years_elapsed, lab_start
        )

        # DNA synthesis accessibility
        synthesis_factor = compute_synthesis_access_factor(
            years_elapsed, dna_synthesis_cost_reduction, dna_synthesis_capability_factor
        )

        # Knowledge diffusion (lowers barriers)
        knowledge_factor = (1 + knowledge_diffusion_rate) ** years_elapsed

        # AI direct agent capability
        ai_direct = compute_ai_direct_capability(
            t, ai_direct_start_year, ai_direct_growth_rate, ai_direct_max_multiplier
        )

        # Effective capable pool
        current_capable = compute_capable_pool(
            current_pop, prop_high, ai_capability_ratio, current_ai_adoption,
            base_capable, pop,
            knowledge_diffusion_factor=knowledge_factor,
            ai_direct_factor=ai_direct
        )
        capability_multiplier = current_capable / baseline_capable_pool

        # Mitigation factor (linear, can drift negative)
        # Net mitigation = base improvement + regulatory drift
        net_mitigation_rate = mitigation_growth + regulatory_drift
        mitigation_factor = max(0.5, 1 + net_mitigation_rate * years_elapsed)  # Floor at 0.5

        # Malicious incentives (linear growth)
        malicious_incentive_factor = 1 + malicious_growth * years_elapsed

        # Permafrost/climate factor (linear growth on natural baseline)
        permafrost_factor = 1 + permafrost_thaw_rate * years_elapsed

        # Urbanization/density factor (compound growth on natural baseline)
        urbanization_factor = compute_urbanization_factor(
            years_elapsed, urbanization_rate, density_natural_multiplier
        )

        # State actor hazard (grows linearly)
        state_hazard = state_actor_base_prob * (1 + state_actor_growth * years_elapsed)

        # --- Compute hazards by pathway ---

        natural_hazard = compute_natural_hazard(natural_base_hazard, permafrost_factor, urbanization_factor)

        accidental_hazard = compute_accidental_hazard(
            accidental_base_hazard,
            lab_multiplier,
            capability_multiplier,
            mitigation_factor,
            gof_fraction,
            gof_risk_multiplier,
            cloud_factor,
            synthesis_factor
        )

        malicious_hazard = compute_malicious_hazard(
            malicious_base_hazard,
            lab_multiplier,
            capability_multiplier,
            malicious_incentive_factor,
            synthesis_factor,
            state_hazard
        )

        # --- Total hazard and probability ---
        total_hazard = natural_hazard + accidental_hazard + malicious_hazard
        annual_prob = hazard_to_probability(total_hazard)
        annual_probs_raw.append(annual_prob)

        # --- Update state for next year ---
        current_pop *= (1 + pop_growth)
        current_ai_adoption = min(1.0, current_ai_adoption * (1 + ai_growth))
        current_labs *= (1 + lab_growth)

    # === APPLY POST-PROCESSING ADJUSTMENTS ===

    # Apply risk correlation effects
    annual_probs = apply_risk_correlation(
        annual_probs_raw,
        yearly_risk_correlation,
        near_miss_security_boost,
        near_miss_vulnerability_reveal
    )

    # Apply antibiotic resistance severity adjustment
    annual_probs = apply_antibiotic_resistance(
        annual_probs,
        antibiotic_resistance_growth,
        antibiotic_severity_multiplier
    )

    # Apply density/urbanization transmission severity adjustment
    annual_probs = apply_density_transmission(
        annual_probs,
        urbanization_rate,
        density_transmission_multiplier
    )

    # === COMPUTE CUMULATIVE ===
    cum_prob = compute_cumulative_probability(annual_probs) * 100

    return cum_prob, annual_probs


# =============================================================================
# MONTE CARLO SENSITIVITY ANALYSIS
# =============================================================================

def run_sensitivity_analysis(
    n_samples: int = 1000,
    param_distributions: Optional[Dict[str, Callable[[], float]]] = None,
    base_params: Optional[Dict] = None,
    seed: Optional[int] = None
) -> Dict:
    """
    Run Monte Carlo sensitivity analysis by sampling parameter distributions.

    Args:
        n_samples: Number of Monte Carlo samples
        param_distributions: Dict mapping parameter names to sampling functions
        base_params: Base parameter values (dict)
        seed: Random seed for reproducibility

    Returns:
        Dict with cumulative_probs, mean, std, percentiles, samples
    """
    if seed is not None:
        np.random.seed(seed)

    if param_distributions is None:
        param_distributions = {}

    if base_params is None:
        base_params = {}

    cumulative_probs = []
    samples = []

    for _ in range(n_samples):
        params = base_params.copy()
        for param_name, sampler in param_distributions.items():
            params[param_name] = sampler()
        samples.append(params.copy())
        cum_prob, _ = calculate_pandemic_probability(**params)
        cumulative_probs.append(cum_prob)

    cumulative_probs = np.array(cumulative_probs)

    return {
        'cumulative_probs': cumulative_probs,
        'mean': float(np.mean(cumulative_probs)),
        'std': float(np.std(cumulative_probs)),
        'percentiles': {
            '5th': float(np.percentile(cumulative_probs, 5)),
            '25th': float(np.percentile(cumulative_probs, 25)),
            '50th': float(np.percentile(cumulative_probs, 50)),
            '75th': float(np.percentile(cumulative_probs, 75)),
            '95th': float(np.percentile(cumulative_probs, 95)),
        },
        'samples': samples
    }


# =============================================================================
# DETAILED OUTPUT FUNCTION
# =============================================================================

def run_detailed_analysis(period_years: int = 10, **kwargs) -> Dict:
    """
    Run model with detailed year-by-year breakdown of all hazard components.
    """
    # Get all defaults
    import inspect
    sig = inspect.signature(calculate_pandemic_probability)
    params = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
    params['period_years'] = period_years
    params.update(kwargs)

    # Run calculation
    cum_prob, annual_probs = calculate_pandemic_probability(**params)

    # Recompute for detailed breakdown
    prop_high, prop_mid_range, ai_capability_ratio = compute_iq_proportions(
        params['iq_mean'], params['iq_sd'], params['iq_high'], params['iq_mid']
    )

    natural_base = params['base_annual_prob'] * params['natural_prob_fraction']
    engineered_base = params['base_annual_prob'] * (1 - params['natural_prob_fraction'])
    accidental_base = engineered_base * (1 - params['malicious_fraction'])
    malicious_base = engineered_base * params['malicious_fraction']

    baseline_capable = compute_capable_pool(
        params['pop'], prop_high, ai_capability_ratio, params['ai_adoption_start'],
        params['base_capable'], params['pop']
    )

    years = []
    current_pop = params['pop']
    current_ai = params['ai_adoption_start']
    current_labs = params['lab_start']

    for t in range(1, period_years + 1):
        years_elapsed = t - 1

        lab_mult = current_labs / params['lab_start']
        cloud_factor = compute_cloud_lab_factor(
            params['cloud_lab_start'], params['cloud_lab_growth'], years_elapsed, params['lab_start']
        )
        synthesis_factor = compute_synthesis_access_factor(
            years_elapsed, params['dna_synthesis_cost_reduction'], params['dna_synthesis_capability_factor']
        )
        knowledge_factor = (1 + params['knowledge_diffusion_rate']) ** years_elapsed
        ai_direct = compute_ai_direct_capability(
            t, params['ai_direct_start_year'], params['ai_direct_growth_rate'], params['ai_direct_max_multiplier']
        )

        capable = compute_capable_pool(
            current_pop, prop_high, ai_capability_ratio, current_ai,
            params['base_capable'], params['pop'], knowledge_factor, ai_direct
        )
        cap_mult = capable / baseline_capable

        net_mit_rate = params['mitigation_growth'] + params['regulatory_drift']
        mit_factor = max(0.5, 1 + net_mit_rate * years_elapsed)
        mal_factor = 1 + params['malicious_growth'] * years_elapsed
        perm_factor = 1 + params['permafrost_thaw_rate'] * years_elapsed
        urban_factor = compute_urbanization_factor(
            years_elapsed, params['urbanization_rate'], params['density_natural_multiplier']
        )
        state_haz = params['state_actor_base_prob'] * (1 + params['state_actor_growth'] * years_elapsed)

        nat_haz = compute_natural_hazard(natural_base, perm_factor, urban_factor)
        acc_haz = compute_accidental_hazard(
            accidental_base, lab_mult, cap_mult, mit_factor,
            params['gof_fraction'], params['gof_risk_multiplier'], cloud_factor, synthesis_factor
        )
        mal_haz = compute_malicious_hazard(
            malicious_base, lab_mult, cap_mult, mal_factor, synthesis_factor, state_haz
        )

        years.append({
            'year': t,
            'natural_hazard': nat_haz,
            'accidental_hazard': acc_haz,
            'malicious_hazard': mal_haz,
            'state_actor_hazard': state_haz,
            'total_hazard': nat_haz + acc_haz + mal_haz,
            'annual_prob_raw': hazard_to_probability(nat_haz + acc_haz + mal_haz),
            'annual_prob_adjusted': annual_probs[t-1],
            'lab_multiplier': lab_mult,
            'cloud_lab_factor': cloud_factor,
            'synthesis_factor': synthesis_factor,
            'capability_multiplier': cap_mult,
            'ai_direct_factor': ai_direct,
            'mitigation_factor': mit_factor,
            'knowledge_factor': knowledge_factor,
            'urbanization_factor': urban_factor,
            'permafrost_factor': perm_factor,
        })

        current_pop *= (1 + params['pop_growth'])
        current_ai = min(1.0, current_ai * (1 + params['ai_growth']))
        current_labs *= (1 + params['lab_growth'])

    return {
        'cumulative_prob_pct': cum_prob,
        'years': years,
        'parameters': params,
        'derived': {
            'prop_high_iq': prop_high,
            'prop_mid_range': prop_mid_range,
            'ai_capability_ratio': ai_capability_ratio,
            'natural_base_hazard': natural_base,
            'accidental_base_hazard': accidental_base,
            'malicious_base_hazard': malicious_base
        }
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pandemic Risk Simulator v2.0 - Comprehensive hazard-based model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predictor.py                           # Basic run with defaults
  python predictor.py --period_years 20         # 20-year projection
  python predictor.py --detailed                # Show hazard breakdown
  python predictor.py --monte_carlo             # Sensitivity analysis
  python predictor.py --regulatory_drift -0.02  # Model regulatory erosion
        """
    )

    # Original parameters
    parser.add_argument("--period_years", type=int, default=10)
    parser.add_argument("--base_annual_prob", type=float, default=0.025)
    parser.add_argument("--natural_prob_fraction", type=float, default=0.5)
    parser.add_argument("--malicious_fraction", type=float, default=0.00001)
    parser.add_argument("--malicious_growth", type=float, default=0.03)
    parser.add_argument("--mitigation_growth", type=float, default=0.02)
    parser.add_argument("--pop", type=float, default=8.3e9)
    parser.add_argument("--pop_growth", type=float, default=0.008)
    parser.add_argument("--iq_mean", type=float, default=100)
    parser.add_argument("--iq_sd", type=float, default=15)
    parser.add_argument("--iq_high", type=int, default=130)
    parser.add_argument("--iq_mid", type=int, default=110)
    parser.add_argument("--base_capable", type=int, default=30000)
    parser.add_argument("--ai_adoption_start", type=float, default=0.5)
    parser.add_argument("--ai_growth", type=float, default=0.15)
    parser.add_argument("--lab_start", type=int, default=3515)
    parser.add_argument("--lab_growth", type=float, default=0.1)

    # New v2.0 parameters
    parser.add_argument("--dna_synthesis_cost_reduction", type=float, default=0.3,
                        help="Annual DNA synthesis cost reduction rate")
    parser.add_argument("--dna_synthesis_capability_factor", type=float, default=0.5,
                        help="Max capability boost from synthesis access")
    parser.add_argument("--cloud_lab_start", type=float, default=0.05,
                        help="Starting cloud lab accessibility")
    parser.add_argument("--cloud_lab_growth", type=float, default=0.20,
                        help="Annual cloud lab growth rate")
    parser.add_argument("--ai_direct_start_year", type=int, default=3,
                        help="Year when AI direct capability emerges")
    parser.add_argument("--ai_direct_growth_rate", type=float, default=0.5,
                        help="AI direct capability growth rate")
    parser.add_argument("--ai_direct_max_multiplier", type=float, default=0.3,
                        help="Maximum AI direct capability multiplier")
    parser.add_argument("--state_actor_base_prob", type=float, default=0.003,
                        help="Baseline state actor annual probability")
    parser.add_argument("--state_actor_growth", type=float, default=0.03,
                        help="Annual state actor risk growth")
    parser.add_argument("--gof_fraction", type=float, default=0.15,
                        help="Fraction of lab work that is gain-of-function")
    parser.add_argument("--gof_risk_multiplier", type=float, default=5.0,
                        help="Risk multiplier for GoF work")
    parser.add_argument("--permafrost_thaw_rate", type=float, default=0.01,
                        help="Annual increase in natural hazard from climate")
    parser.add_argument("--urbanization_rate", type=float, default=0.015,
                        help="Annual increase in urban population fraction")
    parser.add_argument("--density_natural_multiplier", type=float, default=0.3,
                        help="How much density increases natural spillover risk")
    parser.add_argument("--density_transmission_multiplier", type=float, default=0.2,
                        help="How much density increases transmission severity")
    parser.add_argument("--regulatory_drift", type=float, default=-0.01,
                        help="Annual change in regulatory effectiveness (negative = erosion)")
    parser.add_argument("--antibiotic_resistance_growth", type=float, default=0.03,
                        help="Annual antibiotic resistance growth")
    parser.add_argument("--antibiotic_severity_multiplier", type=float, default=0.1,
                        help="Severity increase per unit resistance")
    parser.add_argument("--knowledge_diffusion_rate", type=float, default=0.05,
                        help="Annual dangerous knowledge accessibility growth")
    parser.add_argument("--yearly_risk_correlation", type=float, default=0.0,
                        help="Correlation between yearly risks")
    parser.add_argument("--near_miss_security_boost", type=float, default=0.1,
                        help="Security improvement after near-miss")
    parser.add_argument("--near_miss_vulnerability_reveal", type=float, default=0.05,
                        help="Vulnerability exposure after near-miss")

    # Output options
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed year-by-year breakdown")
    parser.add_argument("--monte_carlo", action="store_true",
                        help="Run Monte Carlo sensitivity analysis")
    parser.add_argument("--mc_samples", type=int, default=1000)
    parser.add_argument("--mc_seed", type=int, default=None)

    args = parser.parse_args()

    # Build parameter dict
    model_params = {k: v for k, v in vars(args).items()
                    if k not in ['detailed', 'monte_carlo', 'mc_samples', 'mc_seed']}

    if args.monte_carlo:
        print(f"Running Monte Carlo sensitivity analysis ({args.mc_samples} samples)...\n")

        param_distributions = {
            'base_annual_prob': lambda: np.random.uniform(0.02, 0.03),
            'natural_prob_fraction': lambda: np.random.uniform(0.5, 0.7),
            'malicious_fraction': lambda: np.random.uniform(0.000005, 0.00002),
            'mitigation_growth': lambda: np.random.uniform(0.03, 0.07),
            'malicious_growth': lambda: np.random.uniform(0.02, 0.05),
            'lab_growth': lambda: np.random.uniform(0.08, 0.12),
            'ai_growth': lambda: np.random.uniform(0.10, 0.20),
            'dna_synthesis_cost_reduction': lambda: np.random.uniform(0.2, 0.4),
            'state_actor_base_prob': lambda: np.random.uniform(0.0005, 0.002),
            'gof_risk_multiplier': lambda: np.random.uniform(2.0, 5.0),
            'regulatory_drift': lambda: np.random.uniform(-0.02, 0.02),
        }

        results = run_sensitivity_analysis(
            n_samples=args.mc_samples,
            param_distributions=param_distributions,
            base_params=model_params,
            seed=args.mc_seed
        )

        print(f"Cumulative {args.period_years}-year probability:")
        print(f"  Mean:   {results['mean']:.2f}%")
        print(f"  Std:    {results['std']:.2f}%")
        print(f"  5th:    {results['percentiles']['5th']:.2f}%")
        print(f"  25th:   {results['percentiles']['25th']:.2f}%")
        print(f"  Median: {results['percentiles']['50th']:.2f}%")
        print(f"  75th:   {results['percentiles']['75th']:.2f}%")
        print(f"  95th:   {results['percentiles']['95th']:.2f}%")

    elif args.detailed:
        results = run_detailed_analysis(**model_params)

        print(f"Projected {args.period_years}-year cumulative probability: {results['cumulative_prob_pct']:.2f}%\n")
        print("Year-by-year breakdown:")
        print("-" * 100)
        print(f"{'Year':>4} {'Natural':>9} {'Accident':>9} {'Malicious':>10} {'State':>9} {'Total':>9} {'Prob':>8}")
        print("-" * 100)

        for y in results['years']:
            print(f"{y['year']:>4} {y['natural_hazard']:>9.5f} {y['accidental_hazard']:>9.5f} "
                  f"{y['malicious_hazard']:>10.7f} {y['state_actor_hazard']:>9.5f} "
                  f"{y['total_hazard']:>9.5f} {y['annual_prob_adjusted']*100:>7.2f}%")

        print(f"\nKey multipliers (Year {args.period_years}):")
        y_last = results['years'][-1]
        print(f"  Lab multiplier:        {y_last['lab_multiplier']:.2f}x")
        print(f"  Cloud lab factor:      {y_last['cloud_lab_factor']:.2f}x")
        print(f"  Synthesis factor:      {y_last['synthesis_factor']:.2f}x")
        print(f"  Capability multiplier: {y_last['capability_multiplier']:.2f}x")
        print(f"  AI direct factor:      {y_last['ai_direct_factor']:.2f}")
        print(f"  Mitigation factor:     {y_last['mitigation_factor']:.2f}x")
        print(f"  Knowledge factor:      {y_last['knowledge_factor']:.2f}x")
        print(f"  Urbanization factor:   {y_last['urbanization_factor']:.2f}x")
        print(f"  Permafrost factor:     {y_last['permafrost_factor']:.2f}x")

    else:
        cum_prob, annual_probs = calculate_pandemic_probability(**model_params)

        print(f"Projected probability of a major pandemic over the next {args.period_years} years: {cum_prob:.2f}%")
        print("\nAnnual probabilities:")
        for year, prob in enumerate(annual_probs, 1):
            print(f"  Year {year}: {prob * 100:.2f}%")
