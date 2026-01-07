#!/usr/bin/env python3
"""
Pandemic Risk Simulator (hazard-based)

What this model is:
- A transparent *scenario calculator* that combines (i) a natural-spillover pathway and
  (ii) engineered pathways (accidental + malicious) into a single annual pandemic probability.

What this model is NOT:
- A validated forecast. Parameter defaults are judgmental and should be treated as priors.

Core design choices (keeps original knobs/ideas):
- Keeps the IQ-tail and "AI enables mid-IQ to perform like high-IQ" mechanism.
- Keeps separate natural vs engineered risk split via `natural_prob_fraction`.
- Keeps accidental vs malicious engineered split (50/50 baseline) with distinct growth/mitigation drivers.
- Uses hazard/rate arithmetic for internal coherence:
    annual_prob = 1 - exp(-annual_hazard)
  and hazards add across pathways.

Interpretation notes (minimal reinterpretations; parameter names and defaults preserved):
- `base_annual_prob` sets the baseline total annual pandemic probability *before* year-varying multipliers
  for engineered pathways are applied. Internally it's converted to a baseline hazard h0 = -ln(1-p).
- `malicious_fraction` scales the malicious engineered hazard linearly (attempt intensity), relative to a
  fixed reference value equal to this function's default malicious_fraction.
  This prevents the parameter from "cancelling itself out" and ensures it meaningfully changes results.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union


# Reference value used to scale malicious attempt intensity.
# It is intentionally fixed (the default value in the original model) so that changing
# `malicious_fraction` actually changes the malicious pathway risk.
MALICIOUS_FRACTION_REFERENCE = 1e-5


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _normal_cdf(z: float) -> float:
    """Standard normal CDF using erf (no SciPy dependency)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _prob_to_hazard(p: float) -> float:
    """Convert annual probability to an annual hazard rate."""
    p = _clamp(p, 0.0, 1.0 - 1e-12)
    return -math.log1p(-p)


def _hazard_to_prob(h: float) -> float:
    """Convert annual hazard rate to annual probability."""
    h = max(0.0, h)
    return 1.0 - math.exp(-h)


@dataclass(frozen=True)
class YearState:
    year_index: int  # 1..period_years
    pop: float
    ai_adoption: float
    labs: float


def calculate_pandemic_probability(
    period_years: int = 10,
    base_annual_prob: float = 0.025,  # Based on expert consensus (2-3% annual for major pandemic, slightly higher to account for increasing risks)
    natural_prob_fraction: float = 0.6,  # Majority of historical pandemics are natural; engineered/malicious growing
    malicious_fraction: float = 0.00001,  # 1 in 100,000 capable individuals might attempt malicious acts annually (based on rarity of ideological extremism leading to bio-attempts, e.g., ~50-100 global attempts base from psych trends in terrorism/extremism)
    malicious_growth: float = 0.03,  # Annual increase in malicious incentives due to geopolitical tensions, climate stress, and third-world nation capabilities (e.g., 5-10 states plus eco-activist groups)
    mitigation_growth: float = 0.05,  # Annual mitigation improvement rate (e.g., biosecurity advances)
    pop: float = 8.3e9,
    pop_growth: float = 0.008,
    iq_mean: float = 100,
    iq_sd: float = 15,
    iq_high: int = 130,
    iq_mid: int = 110,
    base_capable: int = 30000,
    ai_adoption_start: float = 0.5,  # Current AI adoption in life sciences (~40-54% from 2024-2025 surveys)
    ai_growth: float = 0.15,  # Annual AI adoption growth rate (aligned with market CAGR trends)
    lab_start: int = 3515,  # Global BSL-3 labs from 2025 data (BSL-4 ~110, but BSL-3 sufficient for much work)
    lab_growth: float = 0.1,
) -> Tuple[float, List[float]]:
    """Compute cumulative and annual pandemic probabilities.

    Returns:
      (cumulative_probability_percent_over_period, annual_probabilities_list)

    The returned annual probabilities are per-year probabilities (0..1).
    """
    if period_years <= 0:
        return 0.0, []

    # --- IQ / capability layer (kept) ---
    z_high = (iq_high - iq_mean) / float(iq_sd)
    prop_high = 1.0 - _normal_cdf(z_high)  # P(IQ >= iq_high)

    z_mid = (iq_mid - iq_mean) / float(iq_sd)
    prop_mid_range = _normal_cdf(z_high) - _normal_cdf(z_mid)  # P(iq_mid <= IQ < iq_high)

    ratio = prop_mid_range / max(prop_high, 1e-18)  # size(mid-range) / size(high tail)

    # Access fraction: fraction of the high-IQ tail assumed practically capable at baseline.
    base_access_frac = base_capable / max(pop * prop_high, 1.0)

    # Baseline capable (high tail) at year 1 (computed via base_access_frac to keep the knob meaningful)
    capable_high_y1 = base_access_frac * pop * prop_high
    ai_adoption_start = _clamp(ai_adoption_start, 0.0, 1.0)
    effective_capable_y1 = capable_high_y1 * (1.0 + ratio * ai_adoption_start)

    # --- Baseline hazard split ---
    h0 = _prob_to_hazard(base_annual_prob)
    natural_prob_fraction = _clamp(natural_prob_fraction, 0.0, 1.0)
    h_nat0 = h0 * natural_prob_fraction
    h_eng0 = h0 * (1.0 - natural_prob_fraction)

    # Keep original "half accidental / half malicious" idea.
    h_acc0 = 0.5 * h_eng0
    h_mal0 = 0.5 * h_eng0

    # --- Simulation ---
    annual_probs: List[float] = []
    survival = 1.0

    current = YearState(year_index=1, pop=pop, ai_adoption=ai_adoption_start, labs=float(lab_start))

    for t in range(1, period_years + 1):
        # Compounded year-over-year factors.
        mitigation_factor = (1.0 + mitigation_growth) ** (t - 1)  # ↑ defenses -> ↓ hazards
        malicious_incentive = (1.0 + malicious_growth) ** (t - 1)  # ↑ incentives -> ↑ hazards

        lab_multiplier = current.labs / max(float(lab_start), 1.0)

        # Effective capable in year t.
        capable_high_t = base_access_frac * current.pop * prop_high
        effective_capable_t = capable_high_t * (1.0 + ratio * current.ai_adoption)
        capable_multiplier = effective_capable_t / max(effective_capable_y1, 1e-12)

        # Make malicious_fraction matter: scale relative to a fixed reference (the original default).
        malicious_attempt_scale = malicious_fraction / max(MALICIOUS_FRACTION_REFERENCE, 1e-18)

        # Hazards by pathway.
        h_nat = h_nat0
        h_acc = h_acc0 * lab_multiplier * capable_multiplier / max(mitigation_factor, 1e-18)
        h_mal = (
            h_mal0
            * capable_multiplier
            * malicious_incentive
            * malicious_attempt_scale
            / max(mitigation_factor, 1e-18)
        )

        h_total = max(0.0, h_nat + h_acc + h_mal)
        p_year = _hazard_to_prob(h_total)

        annual_probs.append(p_year)
        survival *= (1.0 - p_year)

        # Project next year.
        current = YearState(
            year_index=t + 1,
            pop=current.pop * (1.0 + pop_growth),
            ai_adoption=_clamp(current.ai_adoption * (1.0 + ai_growth), 0.0, 1.0),
            labs=current.labs * (1.0 + lab_growth),
        )

    cum_prob_percent = (1.0 - survival) * 100.0
    return cum_prob_percent, annual_probs


# -------------------------
# Optional sensitivity runner
# -------------------------

Distribution = Union[
    Tuple[str, float, float],  # ("uniform", lo, hi)
    Tuple[str, float, float, float],  # ("triangular", lo, mode, hi)
    Tuple[str, float, float],  # ("normal", mean, sd)
    Callable[[random.Random], float],
]


def run_sensitivity(
    num_sims: int = 1000,
    seed: Optional[int] = 0,
    dists: Optional[Dict[str, Distribution]] = None,
    **base_kwargs,
) -> Dict[str, float]:
    """Lightweight Monte Carlo sensitivity runner.

    Args:
      num_sims: number of simulations
      seed: RNG seed (None for non-deterministic)
      dists: dict mapping parameter name -> distribution specification.
             Supported specs:
               ("uniform", lo, hi)
               ("triangular", lo, mode, hi)
               ("normal", mean, sd)
               callable(rng) -> value
      base_kwargs: baseline parameters passed to calculate_pandemic_probability

    Returns:
      Summary dict with p5/p50/p95 (cumulative percent) and mean.
    """
    if num_sims <= 0:
        raise ValueError("num_sims must be positive")

    rng = random.Random(seed)
    dists = dists or {}

    def draw(spec: Distribution) -> float:
        if callable(spec):
            return float(spec(rng))
        kind = spec[0]
        if kind == "uniform":
            _, lo, hi = spec
            return lo + (hi - lo) * rng.random()
        if kind == "triangular":
            _, lo, mode, hi = spec
            return rng.triangular(lo, hi, mode)
        if kind == "normal":
            _, mu, sd = spec
            return rng.gauss(mu, sd)
        raise ValueError(f"Unsupported distribution spec: {spec!r}")

    vals: List[float] = []
    for _ in range(num_sims):
        kwargs = dict(base_kwargs)
        for k, spec in dists.items():
            kwargs[k] = draw(spec)
        cum, _ = calculate_pandemic_probability(**kwargs)
        vals.append(float(cum))

    vals.sort()
    mean = sum(vals) / len(vals)
    p5 = vals[int(0.05 * (len(vals) - 1))]
    p50 = vals[int(0.50 * (len(vals) - 1))]
    p95 = vals[int(0.95 * (len(vals) - 1))]
    return {"mean": mean, "p5": p5, "p50": p50, "p95": p95}


# -------------------------
# Minimal self-tests (optional)
# -------------------------


def _run_self_tests() -> None:
    base = dict(period_years=10)

    cum0, _ = calculate_pandemic_probability(**base)

    cum_more_labs, _ = calculate_pandemic_probability(**base, lab_growth=0.2)
    assert cum_more_labs >= cum0 - 1e-9, "Increasing lab_growth should not reduce cumulative risk."

    cum_more_mit, _ = calculate_pandemic_probability(**base, mitigation_growth=0.20)
    assert cum_more_mit <= cum0 + 1e-9, "Increasing mitigation_growth should not increase cumulative risk."

    cum_more_ai, _ = calculate_pandemic_probability(**base, ai_growth=0.30)
    assert cum_more_ai >= cum0 - 1e-9, "Increasing ai_growth should not reduce cumulative risk."

    cum_more_base, _ = calculate_pandemic_probability(**base, base_annual_prob=0.05)
    assert cum_more_base >= cum0 - 1e-9, "Increasing base_annual_prob should not reduce cumulative risk."

    cum_more_malfrac, _ = calculate_pandemic_probability(**base, malicious_fraction=1e-4)
    assert cum_more_malfrac >= cum0 - 1e-9, "Increasing malicious_fraction should not reduce cumulative risk."

    print("Self-tests passed.")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pandemic Risk Simulator (hazard-based)")
    parser.add_argument("--period_years", type=int, default=10, help="Number of years for the projection")
    parser.add_argument("--base_annual_prob", type=float, default=0.025, help="Base annual probability")
    parser.add_argument("--natural_prob_fraction", type=float, default=0.6, help="Fraction of base probability that's natural")
    parser.add_argument("--malicious_fraction", type=float, default=0.00001, help="Annual attempt intensity among capable individuals")
    parser.add_argument("--malicious_growth", type=float, default=0.03, help="Annual growth in malicious incentives")
    parser.add_argument("--mitigation_growth", type=float, default=0.05, help="Annual mitigation improvement rate")
    parser.add_argument("--pop", type=float, default=8.3e9, help="Starting global population")
    parser.add_argument("--pop_growth", type=float, default=0.008, help="Annual population growth rate")
    parser.add_argument("--iq_mean", type=float, default=100, help="IQ distribution mean")
    parser.add_argument("--iq_sd", type=float, default=15, help="IQ distribution standard deviation")
    parser.add_argument("--iq_high", type=int, default=130, help="High IQ threshold")
    parser.add_argument("--iq_mid", type=int, default=110, help="Mid IQ threshold with AI")
    parser.add_argument("--base_capable", type=int, default=30000, help="Baseline number capable without AI (high-tail)")
    parser.add_argument("--ai_adoption_start", type=float, default=0.5, help="Starting AI adoption rate")
    parser.add_argument("--ai_growth", type=float, default=0.15, help="Annual AI adoption growth rate")
    parser.add_argument("--lab_start", type=int, default=3515, help="Starting number of relevant labs")
    parser.add_argument("--lab_growth", type=float, default=0.1, help="Annual lab growth rate")
    parser.add_argument("--run_tests", action="store_true", help="Run built-in sanity tests and exit")
    return parser


if __name__ == "__main__":
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.run_tests:
        _run_self_tests()
        raise SystemExit(0)

    cum_prob, annual_probs = calculate_pandemic_probability(
        period_years=args.period_years,
        base_annual_prob=args.base_annual_prob,
        natural_prob_fraction=args.natural_prob_fraction,
        malicious_fraction=args.malicious_fraction,
        malicious_growth=args.malicious_growth,
        mitigation_growth=args.mitigation_growth,
        pop=args.pop,
        pop_growth=args.pop_growth,
        iq_mean=args.iq_mean,
        iq_sd=args.iq_sd,
        iq_high=args.iq_high,
        iq_mid=args.iq_mid,
        base_capable=args.base_capable,
        ai_adoption_start=args.ai_adoption_start,
        ai_growth=args.ai_growth,
        lab_start=args.lab_start,
        lab_growth=args.lab_growth,
    )

    print(
        f"Projected probability of a 5% fatal COVID-like variant or similar pandemic over the next "
        f"{args.period_years} years: {cum_prob:.2f}%"
    )
    print("\nAnnual probabilities:")
    for year, prob in enumerate(annual_probs, 1):
        print(f"Year {year}: {prob * 100:.2f}%")
