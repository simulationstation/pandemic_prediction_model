# Pandemic Prediction Model

A comprehensive simulation framework for pandemic risk assessment, including both historical forecast exploration and forward-looking risk modeling.

## Pandemic Risk Simulator (predictor/)

**DISCLAIMER: This is a simulation model for research and educational purposes only. It is NOT a validated forecast and should NOT be used for policy decisions. Parameter defaults are illustrative and have not been empirically validated.**

### Simulation Results (v2.0 - Aggressive Scenario)

| Time Horizon | Cumulative Probability |
|--------------|------------------------|
| 5-year       | **~24%**               |
| 10-year      | **~59%**               |
| 15-year      | **~88%**               |
| 20-year      | **~99%**               |

*Note: These projections use aggressive assumptions reflecting rapid AI advancement, regulatory erosion, and accelerating risk factors. See "Aggressive Parameters" below.*

### Model Structure

The simulator uses a hazard-based framework with multiple risk pathways:

**Natural Pathways:**
- Zoonotic spillover (baseline)
- Climate-enhanced (permafrost thaw, habitat destruction)

**Accidental Pathways:**
- General lab accidents
- Gain-of-function accidents (3x risk multiplier)
- Cloud lab incidents (less oversight)

**Malicious Pathways:**
- Individual actors (capability + incentive driven)
- State-sponsored programs

### Key Risk Factors Modeled

| Factor | Description |
|--------|-------------|
| DNA synthesis democratization | Costs dropping ~30%/year, bypasses lab access |
| Cloud/remote labs | New access vector with less oversight |
| AI as direct agent | AI designing pathogens without human expertise |
| State-sponsored programs | Separate pathway with different resources |
| Gain-of-function research | Higher-risk subset of lab work |
| Permafrost/climate effects | Growing natural baseline |
| Regulatory drift | Can model both improvement and erosion |
| Antibiotic resistance | Increases pandemic severity |
| Knowledge diffusion | Published research lowers capability barriers |
| Risk correlation | Near-miss effects on subsequent years |

### Usage

```bash
# Basic run (10-year default)
python predictor/predictor.py

# Custom time horizon
python predictor/predictor.py --period_years 20

# Detailed hazard breakdown
python predictor/predictor.py --detailed

# Monte Carlo sensitivity analysis
python predictor/predictor.py --monte_carlo --mc_samples 1000

# Model regulatory erosion scenario
python predictor/predictor.py --regulatory_drift -0.02
```

### Aggressive Parameters (v2.0 defaults)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `natural_prob_fraction` | 50% | Equal natural/engineered risk split |
| `mitigation_growth` | 2%/yr | Slow biosecurity improvement |
| `regulatory_drift` | -1%/yr | Slight regulatory erosion |
| `ai_direct_start_year` | 3 years | AI capability emerges soon |
| `state_actor_base_prob` | 0.3%/yr | Higher state program risk |
| `gof_fraction` | 15% | More gain-of-function work |
| `gof_risk_multiplier` | 5x | GoF significantly riskier |

### Sample Output (10-year, detailed)

```
Year   Natural  Accident  Malicious     State     Total     Prob
   1   0.01250   0.02046  0.0030001   0.00300   0.03596    3.53%
   5   0.01300   0.05959  0.0033604   0.00336   0.07595    7.38%
  10   0.01363   0.13898  0.0038109   0.00381   0.15642   14.82%

Key multipliers (Year 10):
  Lab multiplier:        2.36x
  Synthesis factor:      1.48x
  Capability multiplier: 3.11x
  Mitigation factor:     1.09x  (eroded from 1.45x due to regulatory drift)
```

### Limitations

- Parameter defaults are illustrative, not empirically validated
- Assumes pathway independence (partially addressed by correlation factor)
- Does not model pandemic severity separately
- State actor modeling is simplified
- Climate effects are rough approximations

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
