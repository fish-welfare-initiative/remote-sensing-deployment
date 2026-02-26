# Saved XGBoost water-parameter models

## Files

- `model_registry.json` — list of models + required inputs

- `xgb_nh3.joblib` — NH3 model bundle

- `xgb_chla.joblib` — Chla model bundle

- `xgb_do.joblib` — DO model bundle

- `xgb_ph.joblib` — pH model bundle


## How to use (minimal)

```python
import joblib, pandas as pd
import numpy as np

bundle = joblib.load('saved_models/xgb_ph.joblib')  # example
model = bundle['model']
raw_inputs = bundle['raw_inputs']
feature_names = bundle['feature_names']
transform = bundle['transform']

# df_new must contain the raw input columns listed in bundle['raw_inputs']
X = pd.get_dummies(df_new.reindex(columns=raw_inputs), drop_first=False)
X = X.reindex(columns=feature_names, fill_value=0)

# predict using best_iteration if available
bi = bundle.get('best_iteration', None)
if bi is not None:
    try: yhat = model.predict(X, iteration_range=(0, bi+1))
    except TypeError: yhat = model.predict(X)
else:
    yhat = model.predict(X)

if transform == 'log1p':
    yhat = np.expm1(yhat)
```

## Required inputs per model

### NH3 (xgb_nh3.joblib)
- Raw input columns:
  - `s2_cloud_pct`
  - `cp_probability`
  - `s2_prop_MEAN_SOLAR_AZIMUTH_ANGLE`
  - `s2_prop_MEAN_SOLAR_ZENITH_ANGLE`
  - `s2_prop_SOLAR_IRRADIANCE_B1`
  - `s2_prop_MEAN_INCIDENCE_AZIMUTH_ANGLE_B1`
  - `s2_prop_MEAN_INCIDENCE_ZENITH_ANGLE_B1`
  - `RCC`
  - `s2_med3_B10_median`
  - `s2_med3_B11_median`
  - `s2_med3_B12_median`
  - `s2_med3_B1_median`
  - `s2_med3_B2_median`
  - `s2_med3_B3_median`
  - `s2_med3_B4_median`
  - `s2_med3_B5_median`
  - `s2_med3_B6_median`
  - `s2_med3_B7_median`
  - `s2_med3_B8A_median`
  - `s2_med3_B8_median`
  - `s2_med3_B9_median`
  - `s2_med3_probability_median`
  - `s2_med3_NDWI`
  - `cp_source`
  - `t2m_3d`
  - `rh_3d`
  - `cloud_3d`
  - `ws10_3d`
  - `precip_3d_sum`
  - `swrad_3d_sum`
  - `sunshine_3d_sum_s`
  - `t2m_inst`
  - `rh_inst`
  - `cloud_inst`
  - `ws10_inst`
  - `swrad_inst`
  - `precip_inst`
  - `sunshine_inst_s`
  - `weather_code_3d_mode`
  - `t2m_3d_mean`
  - `rh_3d_mean`
  - `cloud_3d_mean`
  - `ws10_3d_mean`

### Chla (xgb_chla.joblib)
- Raw input columns:
  - `s2_cloud_pct`
  - `cp_probability`
  - `s2_prop_MEAN_SOLAR_AZIMUTH_ANGLE`
  - `s2_prop_MEAN_SOLAR_ZENITH_ANGLE`
  - `s2_prop_SOLAR_IRRADIANCE_B1`
  - `s2_prop_MEAN_INCIDENCE_AZIMUTH_ANGLE_B1`
  - `s2_prop_MEAN_INCIDENCE_ZENITH_ANGLE_B1`
  - `RCC`
  - `s2_med3_B10_median`
  - `s2_med3_B11_median`
  - `s2_med3_B12_median`
  - `s2_med3_B1_median`
  - `s2_med3_B2_median`
  - `s2_med3_B3_median`
  - `s2_med3_B4_median`
  - `s2_med3_B5_median`
  - `s2_med3_B6_median`
  - `s2_med3_B7_median`
  - `s2_med3_B8A_median`
  - `s2_med3_B8_median`
  - `s2_med3_B9_median`
  - `s2_med3_probability_median`
  - `s2_med3_NDWI`
  - `cp_source`
  - `t2m_3d`
  - `rh_3d`
  - `cloud_3d`
  - `ws10_3d`
  - `precip_3d_sum`
  - `swrad_3d_sum`
  - `sunshine_3d_sum_s`
  - `t2m_inst`
  - `rh_inst`
  - `cloud_inst`
  - `ws10_inst`
  - `swrad_inst`
  - `precip_inst`
  - `sunshine_inst_s`
  - `weather_code_3d_mode`
  - `t2m_3d_mean`
  - `rh_3d_mean`
  - `cloud_3d_mean`
  - `ws10_3d_mean`

### DO (xgb_do.joblib)
- Raw input columns:
  - `s2_cloud_pct`
  - `cp_probability`
  - `s2_prop_MEAN_SOLAR_AZIMUTH_ANGLE`
  - `s2_prop_MEAN_SOLAR_ZENITH_ANGLE`
  - `s2_prop_SOLAR_IRRADIANCE_B1`
  - `s2_prop_MEAN_INCIDENCE_AZIMUTH_ANGLE_B1`
  - `s2_prop_MEAN_INCIDENCE_ZENITH_ANGLE_B1`
  - `RCC`
  - `s2_med3_B10_median`
  - `s2_med3_B11_median`
  - `s2_med3_B12_median`
  - `s2_med3_B1_median`
  - `s2_med3_B2_median`
  - `s2_med3_B3_median`
  - `s2_med3_B4_median`
  - `s2_med3_B5_median`
  - `s2_med3_B6_median`
  - `s2_med3_B7_median`
  - `s2_med3_B8A_median`
  - `s2_med3_B8_median`
  - `s2_med3_B9_median`
  - `s2_med3_probability_median`
  - `s2_med3_NDWI`
  - `cp_source`
  - `t2m_3d`
  - `rh_3d`
  - `cloud_3d`
  - `ws10_3d`
  - `precip_3d_sum`
  - `swrad_3d_sum`
  - `sunshine_3d_sum_s`
  - `t2m_inst`
  - `rh_inst`
  - `cloud_inst`
  - `ws10_inst`
  - `swrad_inst`
  - `precip_inst`
  - `sunshine_inst_s`
  - `weather_code_3d_mode`
  - `t2m_3d_mean`
  - `rh_3d_mean`
  - `cloud_3d_mean`
  - `ws10_3d_mean`

### pH (xgb_ph.joblib)
- Raw input columns:
  - `s2_cloud_pct`
  - `cp_probability`
  - `s2_prop_MEAN_SOLAR_AZIMUTH_ANGLE`
  - `s2_prop_MEAN_SOLAR_ZENITH_ANGLE`
  - `s2_prop_SOLAR_IRRADIANCE_B1`
  - `s2_prop_MEAN_INCIDENCE_AZIMUTH_ANGLE_B1`
  - `s2_prop_MEAN_INCIDENCE_ZENITH_ANGLE_B1`
  - `RCC`
  - `s2_med3_B10_median`
  - `s2_med3_B11_median`
  - `s2_med3_B12_median`
  - `s2_med3_B1_median`
  - `s2_med3_B2_median`
  - `s2_med3_B3_median`
  - `s2_med3_B4_median`
  - `s2_med3_B5_median`
  - `s2_med3_B6_median`
  - `s2_med3_B7_median`
  - `s2_med3_B8A_median`
  - `s2_med3_B8_median`
  - `s2_med3_B9_median`
  - `s2_med3_probability_median`
  - `s2_med3_NDWI`
  - `cp_source`
  - `t2m_3d`
  - `rh_3d`
  - `cloud_3d`
  - `ws10_3d`
  - `precip_3d_sum`
  - `swrad_3d_sum`
  - `sunshine_3d_sum_s`
  - `t2m_inst`
  - `rh_inst`
  - `cloud_inst`
  - `ws10_inst`
  - `swrad_inst`
  - `precip_inst`
  - `sunshine_inst_s`
  - `weather_code_3d_mode`
  - `t2m_3d_mean`
  - `rh_3d_mean`
  - `cloud_3d_mean`
  - `ws10_3d_mean`
