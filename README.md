# Pandemic Prediction Model

A comprehensive simulation framework for pandemic risk assessment, including both historical forecast exploration and forward-looking risk modeling.

---

## Pandemic Risk Simulator (predictor/)

**DISCLAIMER: This is a simulation model for research and educational purposes only. It is NOT a validated forecast and should NOT be used for policy decisions. Parameter defaults are illustrative and have not been empirically validated. The model is intentionally calibrated for pessimistic/aggressive scenarios to support "prepare for the worst" risk planning.**

### Simulation Results (v2.0 - Aggressive Scenario)

| Time Horizon | Cumulative Probability | Annual Prob (Final Year) |
|--------------|------------------------|--------------------------|
| 5-year       | **~24%**               | 7.38%                    |
| 10-year      | **~59%**               | 14.82%                   |
| 15-year      | **~88%**               | 27.51%                   |
| 20-year      | **~99%**               | 49.13%                   |

### Model Architecture

The simulator uses a **hazard-based competing risks framework**. Total annual risk is computed as:

```
P(pandemic in year) = 1 - exp(-total_hazard)
total_hazard = natural_hazard + accidental_hazard + malicious_hazard + state_hazard
```

This is standard survival analysis methodology that properly handles multiple independent risk sources.

#### Risk Pathways

| Pathway | Components | Scales With |
|---------|------------|-------------|
| **Natural** | Zoonotic spillover, climate-enhanced emergence | Permafrost thaw rate |
| **Accidental** | General lab accidents, GoF accidents, cloud lab incidents | Labs × Capability ÷ Mitigation |
| **Malicious (Individual)** | Lone actors, small groups | Capability × Incentives × Synthesis access |
| **Malicious (State)** | Nation-state bioweapons programs | State actor growth rate |

#### Capability Expansion Model

The model uses an IQ-threshold mechanism to estimate the "capable pool" of individuals who could potentially cause a pandemic:

```
capable_pool = base_capable × pop_growth × ai_expansion × knowledge_diffusion + ai_direct
```

Where:
- **Base capable**: ~30,000 individuals with requisite expertise (IQ ≥ 130 threshold)
- **AI expansion**: AI tools enable mid-range individuals (IQ 110-130) to achieve equivalent capability
- **Knowledge diffusion**: Published research lowers barriers over time
- **AI direct**: AI systems that can design pathogens without human expertise

---

### Complete Parameter Reference

#### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `period_years` | 10 | Simulation time horizon (years) |
| `base_annual_prob` | 0.025 (2.5%) | Baseline total annual hazard rate at t=0 |
| `natural_prob_fraction` | 0.50 (50%) | Fraction of base hazard from natural sources |
| `malicious_fraction` | 0.00001 | Fraction of engineered risk from malicious pathway |

#### Population & Capability Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop` | 8.3×10⁹ | Starting global population |
| `pop_growth` | 0.008 (0.8%/yr) | Annual compound population growth rate |
| `iq_mean` | 100 | Population IQ distribution mean |
| `iq_sd` | 15 | Population IQ standard deviation |
| `iq_high` | 130 | IQ threshold for baseline capability (no AI) |
| `iq_mid` | 110 | IQ threshold for AI-assisted capability |
| `base_capable` | 30,000 | Baseline number of capable individuals at t=0 |

#### Lab Infrastructure Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lab_start` | 3,515 | Starting number of BSL-3+ labs globally |
| `lab_growth` | 0.10 (10%/yr) | Annual compound lab growth rate |
| `cloud_lab_start` | 0.05 (5%) | Starting cloud lab accessibility (fraction of traditional) |
| `cloud_lab_growth` | 0.20 (20%/yr) | Annual cloud lab accessibility growth |

#### AI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ai_adoption_start` | 0.50 (50%) | Starting AI adoption rate in life sciences |
| `ai_growth` | 0.15 (15%/yr) | Annual compound AI adoption growth (caps at 100%) |
| `ai_direct_start_year` | 3 | Year when AI direct pathogen design capability emerges |
| `ai_direct_growth_rate` | 0.5 | Sigmoid steepness for AI capability phase transition |
| `ai_direct_max_multiplier` | 0.30 (30%) | Maximum capability addition from AI direct agent |

#### DNA Synthesis Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dna_synthesis_cost_reduction` | 0.30 (30%/yr) | Annual synthesis cost reduction rate |
| `dna_synthesis_capability_factor` | 0.50 (50%) | Max capability boost from synthesis access |

#### Gain-of-Function Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gof_fraction` | 0.15 (15%) | Fraction of lab work that is gain-of-function |
| `gof_risk_multiplier` | 5.0 | Risk multiplier for GoF vs general lab work |

#### State Actor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `state_actor_base_prob` | 0.003 (0.3%/yr) | Baseline annual probability from state programs |
| `state_actor_growth` | 0.03 (3%/yr) | Annual linear growth in state program risk |

#### Defense & Mitigation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mitigation_growth` | 0.02 (2%/yr) | Annual linear growth in biosecurity effectiveness |
| `regulatory_drift` | -0.01 (-1%/yr) | Annual change in regulatory effectiveness (negative = erosion) |

#### Climate & Environmental Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `permafrost_thaw_rate` | 0.01 (1%/yr) | Annual increase in natural hazard from climate effects |

#### Antibiotic Resistance Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `antibiotic_resistance_growth` | 0.03 (3%/yr) | Annual compound growth in resistance |
| `antibiotic_severity_multiplier` | 0.10 (10%) | Severity increase per unit resistance growth |

#### Knowledge & Incentive Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `knowledge_diffusion_rate` | 0.05 (5%/yr) | Annual growth in dangerous knowledge accessibility |
| `malicious_growth` | 0.03 (3%/yr) | Annual linear growth in malicious incentives |

#### Risk Correlation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yearly_risk_correlation` | 0.0 | Correlation factor between consecutive years |
| `near_miss_security_boost` | 0.10 (10%) | Risk reduction multiplier after high-risk year |
| `near_miss_vulnerability_reveal` | 0.05 (5%) | Risk increase from revealed vulnerabilities |

---

### Growth Model Summary

| Growth Type | Parameters | Formula |
|-------------|------------|---------|
| **Compound** | pop, labs, AI adoption, synthesis costs, cloud labs, knowledge, antibiotic resistance | `value × (1 + rate)^t` |
| **Linear** | mitigation, regulatory drift, malicious incentives, permafrost, state actors | `1 + rate × t` |
| **Sigmoid** | AI direct capability | `1 / (1 + exp(-k(t - t₀)))` |

---

### Usage

```bash
# Basic run (10-year default)
python predictor/predictor.py

# Custom time horizon
python predictor/predictor.py --period_years 20

# Detailed hazard breakdown by year
python predictor/predictor.py --detailed

# Monte Carlo sensitivity analysis (1000 samples)
python predictor/predictor.py --monte_carlo --mc_samples 1000

# Reproducible Monte Carlo with seed
python predictor/predictor.py --monte_carlo --mc_samples 1000 --mc_seed 42

# Model severe regulatory erosion scenario
python predictor/predictor.py --regulatory_drift -0.03

# Conservative scenario (slower AI, better mitigation)
python predictor/predictor.py --ai_direct_start_year 8 --mitigation_growth 0.05 --regulatory_drift 0.0

# Extreme scenario
python predictor/predictor.py --state_actor_base_prob 0.005 --gof_risk_multiplier 10 --regulatory_drift -0.02
```

---

### Sample Output

#### Standard Output (10-year)
```
Projected probability of a major pandemic over the next 10 years: 59.18%

Annual probabilities:
  Year 1: 3.53%
  Year 2: 4.15%
  Year 3: 4.97%
  Year 4: 6.02%
  Year 5: 7.38%
  Year 6: 9.10%
  Year 7: 10.27%
  Year 8: 11.59%
  Year 9: 13.10%
  Year 10: 14.82%
```

#### Detailed Output (--detailed flag)
```
Year   Natural  Accident  Malicious     State     Total     Prob
   1   0.01250   0.02046  0.0030001   0.00300   0.03596    3.53%
   2   0.01263   0.02666  0.0030902   0.00309   0.04238    4.15%
   3   0.01275   0.03482  0.0031802   0.00318   0.05075    4.97%
   4   0.01288   0.04553  0.0032703   0.00327   0.06167    6.02%
   5   0.01300   0.05959  0.0033604   0.00336   0.07595    7.38%
   6   0.01313   0.07768  0.0034505   0.00345   0.09426    9.10%
   7   0.01325   0.08986  0.0035406   0.00354   0.10665   10.27%
   8   0.01338   0.10391  0.0036307   0.00363   0.12092   11.59%
   9   0.01350   0.12017  0.0037208   0.00372   0.13739   13.10%
  10   0.01363   0.13898  0.0038109   0.00381   0.15642   14.82%

Key multipliers (Year 10):
  Lab multiplier:        2.36x
  Cloud lab factor:      0.26x
  Synthesis factor:      1.48x
  Capability multiplier: 3.11x
  AI direct factor:      0.29
  Mitigation factor:     1.09x
  Knowledge factor:      1.55x
```

#### Monte Carlo Output (--monte_carlo flag)
```
Running Monte Carlo sensitivity analysis (1000 samples)...

Cumulative 10-year probability:
  Mean:   59.42%
  Std:    7.83%
  5th:    46.21%
  25th:   53.89%
  Median: 59.71%
  75th:   65.12%
  95th:   72.34%
```

---

### Why These Parameters? (Rationale)

#### Aggressive Assumptions (Pessimistic)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `natural_prob_fraction` | 50% | Engineered risk growing faster than natural |
| `mitigation_growth` | 2%/yr | Biosecurity improvements are slow and underfunded |
| `regulatory_drift` | -1%/yr | Political/economic pressures erode oversight |
| `ai_direct_start_year` | 3 years | AI capabilities advancing faster than expected |
| `state_actor_base_prob` | 0.3%/yr | Multiple states with suspected programs |
| `gof_fraction` | 15% | GoF research expanding despite controversy |
| `gof_risk_multiplier` | 5x | Intentionally enhanced pathogens are much riskier |

#### Conservative Alternative Values

For less pessimistic projections, consider:

```bash
python predictor/predictor.py \
  --natural_prob_fraction 0.70 \
  --mitigation_growth 0.05 \
  --regulatory_drift 0.0 \
  --ai_direct_start_year 8 \
  --state_actor_base_prob 0.001 \
  --gof_risk_multiplier 3.0
```

This produces ~30-35% 10-year probability instead of ~59%.

---

### Limitations & Caveats

1. **Not a validated forecast** - Parameter defaults are judgmental estimates, not empirically calibrated
2. **Pathway independence** - Assumes natural/accidental/malicious risks are independent (partially addressed by correlation parameters)
3. **No severity modeling** - Counts any major pandemic equally; doesn't distinguish CFR
4. **Simplified state actor model** - Treats state programs as single hazard rate
5. **IQ-capability proxy** - Uses IQ distribution as simplified proxy for complex skill requirements
6. **Climate uncertainty** - Permafrost/habitat effects are rough approximations
7. **No feedback loops** - Doesn't model how a pandemic would affect subsequent risk (policy changes, infrastructure damage)

---

### Testing

Run the test suite:

```bash
pytest tests/test_predictor.py -v --noconftest
```

Tests cover:
- Monotonicity (increasing labs → increasing risk)
- Parameter sensitivity (all parameters affect output)
- Boundary conditions (extreme values)
- Mathematical correctness (hazard-to-probability conversion)
- Regression tests (malicious_fraction bug fix)

---

## COVID-19 Forecast Hub Explorer (app/)

An open-source tool for exploring, transforming, and serving COVID-19 forecast data from the CDC Forecast Hub.

## Overview

This project provides:

1. **Data Ingestion** - Pull forecast data from the CDC COVID-19 Forecast Hub (S3 or GitHub)
2. **Query Layer** - Python API for querying forecasts with flexible filters
3. **REST API** - FastAPI endpoints for programmatic access
4. **Interactive Dashboard** - Streamlit UI for exploring and visualizing forecasts
5. **Ensemble Support** - Official ensemble retrieval and user-defined ensemble computation

## Disclaimers

**Important:**

- Forecasts are **probabilistic and uncertain** - they represent possible futures, not predictions
- **User-defined transforms** (severity/reporting multipliers, time shifts) are NOT official CDC outputs
- Do **not** use this tool as the sole basis for medical or policy decisions
- Always refer to official CDC guidance for authoritative information

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd pandemic_prediction_model

# Install dependencies
pip install -r requirements.txt

# Or use make
make install
```

### Data Ingestion

Pull forecast data from the CDC COVID-19 Forecast Hub:

```bash
# From S3 (recommended, faster)
python -m app.ingest --source s3 --refresh

# From GitHub (fallback)
python -m app.ingest --source github --refresh

# With verbose logging
python -m app.ingest --source s3 --refresh --verbose

# Limit files for testing
python -m app.ingest --source s3 --limit 10
```

Or using make:

```bash
make ingest-s3
make ingest-github
```

### Run the API

```bash
# Start FastAPI server
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

# Or use make
make api
```

API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Run the Dashboard

```bash
# Start Streamlit
streamlit run app/ui.py --server.port 8501

# Or use make
make ui
```

Dashboard available at: http://localhost:8501

### Run Tests

```bash
pytest tests/ -v

# Or use make
make test
```

## Project Structure

```
pandemic_prediction_model/
├── app/
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration settings
│   ├── schema.py           # Pydantic models & validation
│   ├── storage.py          # DuckDB storage layer
│   ├── ingest.py           # Data ingestion (S3/GitHub)
│   ├── query.py            # Query functions
│   ├── ensemble.py         # Ensemble computation
│   ├── api.py              # FastAPI REST API
│   └── ui.py               # Streamlit dashboard
├── tests/
│   ├── conftest.py         # Pytest fixtures
│   ├── test_schema.py      # Schema tests
│   ├── test_ensemble.py    # Ensemble tests
│   ├── test_storage.py     # Storage tests
│   └── test_api.py         # API tests
├── data_cache/             # Local cache for downloaded files
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration
├── Makefile               # Common commands
└── README.md              # This file
```

## Features

### Data Ingestion

- **S3 Access**: Anonymous access to `s3://covid19-forecast-hub`
- **GitHub Fallback**: Download from `CDCgov/covid19-forecast-hub` repository
- **Smart Caching**: Local parquet/CSV cache with manifest tracking
- **Schema Adaptation**: Handles column name variations across hub versions

### Query Layer

```python
from app.query import get_forecast, list_models, list_locations

# List available data
models = list_models()
locations = list_locations()
targets = list_targets()
ref_dates = list_reference_dates()

# Query forecasts
df = get_forecast(
    target="wk inc covid hosp",
    location="US",
    reference_date=date(2024, 1, 15),
    horizons=[0, 1, 2, 3],
    model_ids=["CovidHub-ensemble"],
    quantiles=[0.025, 0.5, 0.975]
)
```

### Ensemble Support

```python
from app.ensemble import (
    get_official_ensemble,
    compute_median_ensemble,
    official_or_best_ensemble
)

# Get official CDC ensemble
official = get_official_ensemble(
    target="wk inc covid hosp",
    location="US",
    reference_date=date(2024, 1, 15)
)

# Compute custom ensemble from selected models
user_ensemble = compute_median_ensemble(
    target="wk inc covid hosp",
    location="US",
    reference_date=date(2024, 1, 15),
    model_ids=["Model-A", "Model-B", "Model-C"]
)

# Auto-select best available
ensemble, ens_type = official_or_best_ensemble(...)
```

### User-Defined Transforms

Apply scenario transformations (clearly marked as non-official):

```python
from app.schema import TransformParams
from app.ensemble import apply_transform

transform = TransformParams(
    severity_multiplier=1.5,    # 0.25 - 4.0
    reporting_multiplier=1.2,   # 0.5 - 2.0
    time_shift_weeks=1          # -2 to +2
)

transformed_df = apply_transform(forecast_df, transform)
```

### REST API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /stats` | Database statistics |
| `GET /meta/targets` | List available targets |
| `GET /meta/locations` | List available locations |
| `GET /meta/models` | List available models |
| `GET /meta/reference-dates` | List reference dates |
| `GET /meta/horizons` | List available horizons |
| `GET /meta/ensemble` | Ensemble model info |
| `GET /forecast` | Query forecasts |
| `GET /forecast/transformed` | Get transformed forecasts |
| `GET /forecast/ensemble` | Get ensemble forecasts |
| `GET /forecast/summary` | Forecast summary |
| `GET /observations` | Get observed values |
| `GET /latest` | Get latest reference date |

### Dashboard Pages

1. **Explore Forecasts** - Interactive visualization with adjustable knobs
2. **Compare Models** - Side-by-side model comparison
3. **Build User Ensemble** - Create custom ensembles

## Configuration

Environment variables (prefix `FORECAST_HUB_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `S3_BUCKET` | `covid19-forecast-hub` | S3 bucket name |
| `S3_PREFIX` | `model-output` | S3 prefix for model outputs |
| `DATABASE_PATH` | `./forecast_hub.duckdb` | DuckDB database path |
| `DATA_CACHE_DIR` | `./data_cache` | Local cache directory |
| `API_PORT` | `8000` | FastAPI server port |
| `STREAMLIT_PORT` | `8501` | Streamlit dashboard port |

## Data Attribution

This tool uses data from the [CDC COVID-19 Forecast Hub](https://github.com/CDCgov/covid19-forecast-hub):

- Primary source: `s3://covid19-forecast-hub`
- GitHub mirror: `https://github.com/CDCgov/covid19-forecast-hub`

The forecast hub collects probabilistic nowcasts/forecasts from multiple modeling teams. Data is in "hubverse" format with task-id columns and output representation columns.

**If publishing analysis based on this data:**
- Please contact the hub maintainers
- Cite the contributing models appropriately
- Follow any data use guidelines from the hub

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test file
pytest tests/test_schema.py -v
```

### Code Quality

```bash
# Type checking (if mypy installed)
mypy app/

# Formatting (if black installed)
black app/ tests/

# Linting (if ruff installed)
ruff check app/ tests/
```

## Technical Details

### Storage

- **DuckDB**: Local analytical database for fast queries
- **Parquet Cache**: Compressed file cache for downloaded data
- **Manifest**: JSON tracking of cached files and timestamps

### Schema Handling

The system adapts to column name variations:
- `reference_date` / `origin_date` / `forecast_date`
- `location` / `FIPS` / `geo_value`
- `horizon` / `weeks_ahead`
- etc.

### Validation

- Required columns: `model_id`, `output_type`, `output_type_id`, `value`, `location`, `reference_date`, `horizon`, `target`
- Output types: `quantile`, `median`, `mean`, `sample`, `cdf`, `pmf`
- Quantile values must be in (0, 1)
- Forecast values must be numeric

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and feature requests, please use the GitHub issue tracker.
