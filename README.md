# Remote Sensing Deployment — Chl-a Prediction Pipeline

This repository contains a standalone, deployable version of Sol White's Sentinel-2 + weather → Chl-a prediction pipeline. It was extracted from Sol's original Colab notebook (`S2_downloadand_climate.ipynb`) and packaged into a single Python script that can be run locally or on any server with Google Earth Engine access.

**Purpose:** Run Sol's trained XGBoost Chl-a model on new pond/date combinations by querying live Sentinel-2 imagery and Open-Meteo weather data.

## Repository structure

```
├── predict_chla.py                  # Main pipeline script (extracted from Sol's notebook)
├── models/                          # Sol's trained models (UNCHANGED from his delivery)
│   ├── xgb_chla.joblib              # XGBoost Chl-a model bundle
│   ├── xgb_do.joblib                # XGBoost DO model bundle
│   ├── xgb_nh3.joblib               # XGBoost NH3 model bundle
│   ├── xgb_ph.joblib                # XGBoost pH model bundle
│   ├── model_registry.json          # Model metadata (features, transforms, best_iteration)
│   └── README.md                    # Sol's original model documentation
├── results/                         # Output from the IC.v2 prediction run
│   ├── IC_v2_Chla_Predictions.xlsx  # Summary table with chart
│   └── IC_v2_full_features_and_predictions.csv  # All 56 feature columns + prediction
├── 2026 Github ARA Pond IDs Key.csv # Pond ID → GPS coordinate lookup
└── README.md                        # This file
```

## How it works

The pipeline has three stages, matching Sol's notebook cells:

### 1. Weather data (Open-Meteo API)
*Source: Sol's notebook Cell 3 — `openmeteo_range_join()`*

For each site/datetime, fetches from Open-Meteo:
- **3-day rolling aggregates** (D−3 to D−1): temperature mean, relative humidity mean, cloud cover mean, wind speed mean, precipitation sum, shortwave radiation sum, sunshine duration sum, and weather code mode
- **Instantaneous values** at the sample hour: temperature, relative humidity, cloud cover, wind speed, shortwave radiation, precipitation, sunshine duration

Backend routing: uses the Archive API for dates before 2022, Historical Forecast API for 2022+, and Forecast API for recent dates.

Output columns: `t2m_3d`, `rh_3d`, `cloud_3d`, `ws10_3d`, `precip_3d_sum`, `swrad_3d_sum`, `sunshine_3d_sum_s`, `t2m_inst`, `rh_inst`, `cloud_inst`, `ws10_inst`, `swrad_inst`, `precip_inst`, `sunshine_inst_s`, `weather_code_3d_mode`, `t2m_3d_mean`, `rh_3d_mean`, `cloud_3d_mean`, `ws10_3d_mean`

### 2. Sentinel-2 extraction (Google Earth Engine)
*Source: Sol's notebook Cells 5–6 — `find_closest_s2_and_clip()`, `s2_attach_cloudprob_clipped()`, `process_fast_clipped()`*

For each site/datetime:
1. Queries `COPERNICUS/S2_HARMONIZED` (L1C) within ±1 day, finds the closest image in time
2. Clips to a 100 m buffer around the pond coordinate
3. Attaches cloud probability from `COPERNICUS/S2_CLOUD_PROBABILITY` (matched by `system:index`, falling back to nearest-in-time)
4. Divides L1C reflectance values by 10,000 (standard scaling)
5. Computes **3×3 median** (using `reduceNeighborhood` with `ee.Kernel.square(1)`) for all 13 bands + cloud probability
6. Extracts single-pixel cloud probability at the exact point
7. Extracts image properties: solar angles, irradiance, REFLECTANCE_CONVERSION_CORRECTION

Derived features:
- `s2_med3_NDWI` = (B3_median − B8_median) / (B3_median + B8_median)
- `RCC` = REFLECTANCE_CONVERSION_CORRECTION from image metadata

Output columns: `s2_med3_B1_median` through `s2_med3_B12_median`, `s2_med3_B8A_median`, `s2_med3_probability_median`, `s2_med3_NDWI`, `s2_cloud_pct`, `cp_probability`, `cp_source`, `RCC`, and `s2_prop_*` image properties.

### 3. XGBoost prediction
*Source: Sol's `models/run_models.py` and `models/README.md`*

Loads `xgb_chla.joblib`, which contains:
- Trained XGBRegressor
- `raw_inputs`: 40 feature column names
- `feature_names`: 43 one-hot-encoded feature names (includes `cp_source` dummies)
- `best_iteration`: 203
- `transform`: "none" (no log transform for Chl-a)

The prediction step:
1. Selects the 40 raw input columns from the feature DataFrame
2. One-hot encodes categorical columns (`cp_source`) via `pd.get_dummies`
3. Aligns to the exact 43 feature columns the model expects (fills missing with 0)
4. Predicts using `model.predict(X, iteration_range=(0, 204))`

## Known differences from Sol's notebook

**Sol should review these carefully:**

| Aspect | Sol's notebook | This script | Impact |
|--------|---------------|-------------|--------|
| S2 search window | `days_window=0.125` (±3 hours) | `days_window=1` (±1 day) | Wider window ensures we find same-day S2 passes even if the query time doesn't exactly match the 05:14 UTC flyover. Sol's notebook used ±3h because the input data had exact sample timestamps; here we only know the date. |
| Input timestamps | Exact field sampling times (local time converted to UTC) | Estimated at 05:14 UTC (approximate S2 flyover time for Andhra Pradesh) | Minor — the S2 image selected is the same since there's only one per day. Weather instantaneous values may differ slightly if they were matched to a different hour. |
| Weather API | Open-Meteo (same code) | Open-Meteo (same code) | Should be identical. |
| Band scaling | L1C / 10000 | L1C / 10000 | Identical. |
| 3×3 median | `ee.Kernel.square(1)` | `ee.Kernel.square(1)` | Identical. |
| Cloud probability matching | Index match, then time-nearest | Index match, then time-nearest | Identical logic. |
| Model files | Original `.joblib` bundles | Copied unchanged | Identical — SHA checksums should match. |
| `run_models.py` | Original from Sol | Not used directly — logic reproduced in `predict_chla.py` | Same algorithm: `get_dummies` → `reindex` → `predict(iteration_range)` → `expm1` if log1p. |

## Usage

### Prerequisites

```bash
pip install earthengine-api xgboost joblib pandas numpy httpx nest_asyncio tqdm
```

### GEE authentication

```bash
python -c "import ee; ee.Authenticate(auth_mode='notebook')"
```

### Running predictions

```python
from predict_chla import predict_chla

sites = [
    {"lat": 16.665721, "lon": 81.138444, "datetime": "2025-06-28 05:14:00"},
    {"lat": 16.653449, "lon": 81.147897, "datetime": "2025-07-08 05:14:00"},
]

result = predict_chla(sites)
print(result[["lat", "lon", "pred_Chla"]])
```

### IC.v2 run (what was done)

The IC.v2 validation run covered 19 pond/date combinations from the Innovation 2 dataset (20 were specified but WG-SRI2 lacked GPS coordinates in the ARA key). Pond IDs were matched to GPS coordinates using `2026 Github ARA Pond IDs Key.csv`. Results are in `results/`.

### ARA Pond IDs Key — coordinate mismatches found

When comparing our pipeline output against Sol's IC.v2 validation results, the predicted Chl-a VALUES were all correct but several were assigned to the wrong ponds. The root cause is that the `2026 Github ARA Pond IDs Key.csv` file has GPS coordinates mapped to incorrect pond IDs for 4 groups of ponds:

| ARA Key Pond ID | ARA Key GPS coords actually belong to | Evidence |
|-----------------|---------------------------------------|----------|
| WG-NSR1 | WG-NSR3 | ARA's "NSR1" coords produce Sol's NSR3 prediction (148.30) |
| WG-NSR2 | WG-NSR1 | ARA's "NSR2" coords produce Sol's NSR1 prediction (118.90) |
| WG-NSR3 | WG-NSR2 | ARA's "NSR3" coords produce Sol's NSR2 prediction (135.15) |
| WG-NRR1 | WG-NRR2 | ARA's "NRR1" coords produce Sol's NRR2 prediction (127.55) |
| WG-NRR2 | WG-NRR1 | ARA's "NRR2" coords produce Sol's NRR1 prediction (65.10) |
| WG-AKR1 | WG-AKR2 | ARA's "AKR1" coords produce Sol's AKR2 prediction (95.35) |
| WG-AKR2 | WG-AKR1 | ARA's "AKR2" coords produce Sol's AKR1 prediction (74.22) |
| WG-SRI1 | WG-SRI2 | ARA's "SRI1" coords produce Sol's SRI2 prediction (78.63) |

WG-SRI1's real GPS coordinates are not present in the ARA key at all (and WG-SRI2 has no entry). The corrected results file (`IC_v2_Chla_Predictions_corrected.xlsx`) reassigns predictions to the correct pond IDs based on this analysis. After correction, all 19 predictions match Sol's validation output exactly (18 with correct pond IDs + SRI2 relabeled).

**Sol should verify**: Are the GPS coordinates in the ARA key known to have these swaps? Or did Sol use different GPS sources for his original run?

## Model performance context

From Sol's Remote Sensing Final Report, the Chl-a model on the Innovation 2 validation set achieved:
- R² ≈ 0.59
- RMSE ≈ 47.7 µg/L
- MAE ≈ 35.3 µg/L

Sol noted that Chl-a was the strongest of the four models, benefiting from large external training datasets and the fact that chlorophyll-a is optically active and well-captured by Sentinel-2 bands.
