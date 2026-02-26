#!/usr/bin/env python3
"""
FWI Chl-a Prediction Web App
Flask backend serving the prediction API and frontend.

Usage:
    pip install flask httpx nest_asyncio pandas numpy joblib earthengine-api xgboost
    python app.py

    Then open http://localhost:5000 in your browser.

    Requires:
    - Google Earth Engine credentials (run `earthengine authenticate` first)
    - Model file at ../models/xgb_chla.joblib
    - ARA key file at ../2026 Github ARA Pond IDs Key.csv
"""
import os, sys, json, asyncio, traceback
import numpy as np
import pandas as pd
import httpx
import ee
import joblib
import nest_asyncio
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime, timedelta
from pathlib import Path

nest_asyncio.apply()

app = Flask(__name__, static_folder="static")

# ── Paths (relative to this file) ──
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(BASE_DIR / "models" / "xgb_chla.joblib")
ARA_KEY_PATH = str(BASE_DIR / "2026 Github ARA Pond IDs Key.csv")

# ── Load pond data at startup ──
PONDS_DF = pd.read_csv(ARA_KEY_PATH)
PONDS_WITH_GPS = PONDS_DF[PONDS_DF["latitude"].notna()].copy()

# ── Load model at startup ──
BUNDLE = joblib.load(MODEL_PATH)
MODEL = BUNDLE["model"]
RAW_INPUTS = BUNDLE["raw_inputs"]
FEATURE_NAMES = BUNDLE["feature_names"]
TRANSFORM = BUNDLE.get("transform", "none")
BEST_ITERATION = BUNDLE.get("best_iteration", None)

# ── GEE init ──
EE_INITIALIZED = False

def init_ee():
    global EE_INITIALIZED
    if not EE_INITIALIZED:
        ee.Initialize(project="")
        EE_INITIALIZED = True
        print("[GEE] Initialized")

# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/ponds")
def get_ponds():
    ponds = []
    for _, r in PONDS_WITH_GPS.iterrows():
        ponds.append({
            "id": r["internal_pond_id"],
            "public_id": r["public_pond_id"],
            "region": r["region"],
            "lat": round(r["latitude"], 6),
            "lon": round(r["longitude"], 6),
            "village": r["village"] if pd.notna(r["village"]) else "",
            "farmer": r["farmer_name"] if pd.notna(r["farmer_name"]) else "",
        })
    return jsonify(ponds)

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        pond_id = data["pond_id"]
        date_str = data["date"]  # YYYY-MM-DD

        pond_row = PONDS_WITH_GPS[PONDS_WITH_GPS["internal_pond_id"] == pond_id]
        if pond_row.empty:
            return jsonify({"error": f"Pond {pond_id} not found or has no GPS"}), 400

        lat = float(pond_row.iloc[0]["latitude"])
        lon = float(pond_row.iloc[0]["longitude"])
        target_date = pd.Timestamp(date_str, tz="UTC")

        init_ee()

        # Step 1: Find most recent S2 image BEFORE target date
        s2_result = find_recent_s2(lon, lat, target_date)
        if s2_result is None:
            return jsonify({"error": "No Sentinel-2 image found in the 30 days before this date"}), 404

        s2_data, s2_meta = s2_result

        # Step 2: Get weather for the S2 acquisition date
        s2_date = pd.Timestamp(s2_meta["acq_time"], tz="UTC")
        weather = get_weather(lat, lon, s2_date)

        # Step 3: Build feature row and predict
        features = build_feature_row(s2_data, s2_meta, weather)
        prediction = run_model(features)

        return jsonify({
            "prediction": round(float(prediction), 2),
            "pond": {
                "id": pond_id,
                "lat": lat,
                "lon": lon,
                "region": str(pond_row.iloc[0]["region"]),
                "village": str(pond_row.iloc[0]["village"]) if pd.notna(pond_row.iloc[0]["village"]) else "",
                "farmer": str(pond_row.iloc[0]["farmer_name"]) if pd.notna(pond_row.iloc[0]["farmer_name"]) else "",
            },
            "satellite": {
                "image_id": s2_meta.get("image_id", ""),
                "acq_time": s2_meta.get("acq_time", ""),
                "cloud_pct": s2_meta.get("cloud_pct"),
                "cp_probability": s2_meta.get("cp_probability"),
                "cp_source": s2_meta.get("cp_source", ""),
                "spacecraft": s2_meta.get("spacecraft", ""),
                "days_before_target": round((target_date - s2_date).total_seconds() / 86400, 1),
            },
            "bands": {k: round(v, 6) if v is not None else None for k, v in s2_data.items() if k.startswith("s2_med3_")},
            "ndwi": round(s2_data.get("s2_med3_NDWI", 0), 4),
            "weather": {
                "t2m_3d": weather.get("t2m_3d"),
                "rh_3d": weather.get("rh_3d"),
                "cloud_3d": weather.get("cloud_3d"),
                "ws10_3d": weather.get("ws10_3d"),
                "precip_3d_sum": weather.get("precip_3d_sum"),
                "swrad_3d_sum": weather.get("swrad_3d_sum"),
                "sunshine_3d_sum_s": weather.get("sunshine_3d_sum_s"),
                "t2m_inst": weather.get("t2m_inst"),
                "rh_inst": weather.get("rh_inst"),
                "cloud_inst": weather.get("cloud_inst"),
                "ws10_inst": weather.get("ws10_inst"),
            },
            "feature_importance": get_top_features(features),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# S2 EXTRACTION
# ══════════════════════════════════════════════════════════════

def find_recent_s2(lon, lat, target_date, max_days_back=30):
    """Find the most recent S2 L1C image BEFORE target_date."""
    pt = ee.Geometry.Point(lon, lat)
    aoi = pt.buffer(100)

    t_end = ee.Date(target_date.isoformat())
    t_start = t_end.advance(-max_days_back, "day")

    col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
             .filterBounds(aoi)
             .filterDate(t_start, t_end)
             .sort("system:time_start", False))  # most recent first

    n = int(col.size().getInfo() or 0)
    if n == 0:
        return None

    s2 = ee.Image(col.first()).clip(aoi)

    # Extract properties
    keep = ["system:index", "system:time_start", "CLOUDY_PIXEL_PERCENTAGE",
            "SPACECRAFT_NAME", "MEAN_SOLAR_AZIMUTH_ANGLE", "MEAN_SOLAR_ZENITH_ANGLE",
            "SOLAR_IRRADIANCE_B1", "MEAN_INCIDENCE_AZIMUTH_ANGLE_B1",
            "MEAN_INCIDENCE_ZENITH_ANGLE_B1", "REFLECTANCE_CONVERSION_CORRECTION"]
    props = ee.Dictionary(s2.toDictionary(keep)).getInfo() or {}

    acq_ms = props.get("system:time_start")
    acq_time = ee.Date(acq_ms).format("YYYY-MM-dd HH:mm:ss").getInfo() if acq_ms else ""

    # Cloud probability
    cp_all = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
    s2_idx = ee.String(s2.get("system:index"))
    cp_filt = cp_all.filter(ee.Filter.eq("system:index", s2_idx))
    cp_n = int(cp_filt.size().getInfo() or 0)

    cp_source = "none"
    if cp_n > 0:
        cp_img = ee.Image(cp_filt.first())
        cp_source = "index"
    else:
        cp_time_filt = (cp_all.filterBounds(aoi)
                        .filterDate(t_start, t_end)
                        .map(lambda c: ee.Image(c).set("td", ee.Number(c.get("system:time_start")).subtract(ee.Number(acq_ms)).abs()))
                        .sort("td"))
        if int(cp_time_filt.size().getInfo() or 0) > 0:
            cp_img = ee.Image(cp_time_filt.first())
            cp_source = "time"
        else:
            cp_img = None

    # Attach cloud probability
    bands_to_sample = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]
    scaled = s2.select(bands_to_sample).divide(10000.0)

    if cp_img is not None:
        prob = cp_img.select("probability").clip(aoi)
        work = ee.Image.cat([scaled, prob])
    else:
        work = scaled

    # 3x3 median
    med3 = work.reduceNeighborhood(
        reducer=ee.Reducer.median(),
        kernel=ee.Kernel.square(1)
    )

    vals = med3.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=pt, scale=10, bestEffort=True, maxPixels=1e13
    ).getInfo() or {}

    # Single-pixel CP at point
    cp_at_pt = None
    if cp_img is not None:
        try:
            cp_val = prob.reduceRegion(ee.Reducer.first(), pt, scale=10, bestEffort=True, maxPixels=1e13).getInfo()
            cp_at_pt = list(cp_val.values())[0] if cp_val else None
        except:
            pass

    # Build result dict
    s2_data = {}
    for k, v in vals.items():
        s2_data[f"s2_med3_{k}"] = v

    # NDWI
    b3 = s2_data.get("s2_med3_B3_median")
    b8 = s2_data.get("s2_med3_B8_median")
    if b3 is not None and b8 is not None and (b3 + b8) != 0:
        s2_data["s2_med3_NDWI"] = (b3 - b8) / (b3 + b8)
    else:
        s2_data["s2_med3_NDWI"] = np.nan

    s2_data["s2_cloud_pct"] = props.get("CLOUDY_PIXEL_PERCENTAGE")
    s2_data["cp_probability"] = cp_at_pt
    s2_data["cp_source"] = cp_source
    s2_data["s2_prop_MEAN_SOLAR_AZIMUTH_ANGLE"] = props.get("MEAN_SOLAR_AZIMUTH_ANGLE")
    s2_data["s2_prop_MEAN_SOLAR_ZENITH_ANGLE"] = props.get("MEAN_SOLAR_ZENITH_ANGLE")
    s2_data["s2_prop_SOLAR_IRRADIANCE_B1"] = props.get("SOLAR_IRRADIANCE_B1")
    s2_data["s2_prop_MEAN_INCIDENCE_AZIMUTH_ANGLE_B1"] = props.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_B1")
    s2_data["s2_prop_MEAN_INCIDENCE_ZENITH_ANGLE_B1"] = props.get("MEAN_INCIDENCE_ZENITH_ANGLE_B1")
    s2_data["RCC"] = props.get("REFLECTANCE_CONVERSION_CORRECTION")

    meta = {
        "image_id": props.get("system:index", ""),
        "acq_time": acq_time,
        "cloud_pct": props.get("CLOUDY_PIXEL_PERCENTAGE"),
        "cp_probability": cp_at_pt,
        "cp_source": cp_source,
        "spacecraft": props.get("SPACECRAFT_NAME", ""),
        "candidates": n,
    }

    return s2_data, meta


# ══════════════════════════════════════════════════════════════
# WEATHER
# ══════════════════════════════════════════════════════════════

BASE_FORECAST = "https://api.open-meteo.com/v1/forecast"
BASE_HISTFOR = "https://historical-forecast-api.open-meteo.com/v1/forecast"
BASE_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
CUTOFF_HISTFC = pd.Timestamp(2022, 1, 1, tz="UTC")
RECENT_DAYS_BUF = 5

DAILY_VARS = "temperature_2m_mean,relative_humidity_2m_mean,cloud_cover_mean,wind_speed_10m_mean,precipitation_sum,shortwave_radiation_sum,sunshine_duration,weather_code"
HOURLY_VARS = "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m,shortwave_radiation,sunshine_duration"

def _route_backend(sample_day_utc):
    now_utc = pd.Timestamp.now(tz="UTC")
    cutoff_recent = (now_utc - pd.Timedelta(days=RECENT_DAYS_BUF)).floor("D")
    if sample_day_utc > cutoff_recent:
        return BASE_FORECAST
    if sample_day_utc >= CUTOFF_HISTFC:
        return BASE_HISTFOR
    return BASE_ARCHIVE

def get_weather(lat, lon, s2_date):
    """Get 3-day rolling weather + instantaneous weather for the S2 date."""
    day = s2_date.floor("D")
    start = (day - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    end = day.strftime("%Y-%m-%d")
    end_dminus1 = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    base = _route_backend(day)
    result = {}

    # Daily
    try:
        r = httpx.get(base, params={
            "latitude": lat, "longitude": lon,
            "daily": DAILY_VARS, "start_date": start, "end_date": end_dminus1,
            "timezone": "UTC", "cell_selection": "nearest"
        }, timeout=30)
        js = r.json()
        d = js.get("daily", {})
        dates = pd.to_datetime(d.get("time", []), utc=True).date
        if len(dates) > 0:
            df_d = pd.DataFrame({
                "date": dates,
                "t2m_mean": pd.to_numeric(pd.Series(d.get("temperature_2m_mean", [])), errors="coerce").values,
                "rh_mean": pd.to_numeric(pd.Series(d.get("relative_humidity_2m_mean", [])), errors="coerce").values,
                "cloud_mean": pd.to_numeric(pd.Series(d.get("cloud_cover_mean", [])), errors="coerce").values,
                "ws10_mean": pd.to_numeric(pd.Series(d.get("wind_speed_10m_mean", [])), errors="coerce").values,
                "precip_sum": pd.to_numeric(pd.Series(d.get("precipitation_sum", [])), errors="coerce").values,
                "swrad_sum": pd.to_numeric(pd.Series(d.get("shortwave_radiation_sum", [])), errors="coerce").values,
                "sunshine_s": pd.to_numeric(pd.Series(d.get("sunshine_duration", [])), errors="coerce").values,
                "weather_code": pd.to_numeric(pd.Series(d.get("weather_code", [])), errors="coerce").values,
            }).set_index("date").sort_index()

            roll_mean = df_d[["t2m_mean","rh_mean","cloud_mean","ws10_mean"]].rolling(3, min_periods=3).mean()
            roll_sum = df_d[["precip_sum","swrad_sum","sunshine_s"]].rolling(3, min_periods=3).sum()
            roll_wc = df_d[["weather_code"]].rolling(3, min_periods=3).apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[-1], raw=False)
            roll = pd.concat([roll_mean, roll_sum, roll_wc], axis=1)

            pick = (day - pd.Timedelta(days=1)).date()
            if pick in roll.index:
                v = roll.loc[pick]
                result["t2m_3d"] = float(v.get("t2m_mean", np.nan))
                result["rh_3d"] = float(v.get("rh_mean", np.nan))
                result["cloud_3d"] = float(v.get("cloud_mean", np.nan))
                result["ws10_3d"] = float(v.get("ws10_mean", np.nan))
                result["precip_3d_sum"] = float(v.get("precip_sum", np.nan))
                result["swrad_3d_sum"] = float(v.get("swrad_sum", np.nan))
                result["sunshine_3d_sum_s"] = float(v.get("sunshine_s", np.nan))
                result["weather_code_3d_mode"] = float(v.get("weather_code", np.nan))
                result["t2m_3d_mean"] = result["t2m_3d"]
                result["rh_3d_mean"] = result["rh_3d"]
                result["cloud_3d_mean"] = result["cloud_3d"]
                result["ws10_3d_mean"] = result["ws10_3d"]
    except Exception as e:
        print(f"[Weather] Daily error: {e}")

    # Hourly
    try:
        r = httpx.get(base, params={
            "latitude": lat, "longitude": lon,
            "hourly": HOURLY_VARS, "start_date": start, "end_date": end,
            "timezone": "UTC", "cell_selection": "nearest"
        }, timeout=30)
        js = r.json()
        h = js.get("hourly", {})
        times = pd.to_datetime(h.get("time", []), utc=True)
        if len(times) > 0:
            df_h = pd.DataFrame({"time": times})
            for k in ["temperature_2m","relative_humidity_2m","cloud_cover","wind_speed_10m","shortwave_radiation","precipitation","sunshine_duration"]:
                df_h[k] = pd.to_numeric(pd.Series(h.get(k, [])), errors="coerce").values
            df_h = df_h.set_index("time").sort_index()

            t0 = s2_date.floor("h")
            idxr = df_h.index.get_indexer([t0], method="nearest", tolerance=pd.Timedelta(hours=1))
            if idxr.size and idxr[0] != -1:
                hv = df_h.iloc[idxr[0]]
                result["t2m_inst"] = float(hv.get("temperature_2m", np.nan))
                result["rh_inst"] = float(hv.get("relative_humidity_2m", np.nan))
                result["cloud_inst"] = float(hv.get("cloud_cover", np.nan))
                result["ws10_inst"] = float(hv.get("wind_speed_10m", np.nan))
                result["swrad_inst"] = float(hv.get("shortwave_radiation", np.nan))
                result["precip_inst"] = float(hv.get("precipitation", np.nan))
                result["sunshine_inst_s"] = float(hv.get("sunshine_duration", np.nan))
    except Exception as e:
        print(f"[Weather] Hourly error: {e}")

    return result


# ══════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════

def build_feature_row(s2_data, s2_meta, weather):
    row = {}
    row.update(s2_data)
    row.update(weather)
    df = pd.DataFrame([row])
    X = pd.get_dummies(df.reindex(columns=RAW_INPUTS), drop_first=False)
    X = X.reindex(columns=FEATURE_NAMES, fill_value=0)
    return X

def run_model(X):
    if BEST_ITERATION is not None:
        try:
            yhat = MODEL.predict(X, iteration_range=(0, BEST_ITERATION + 1))
        except TypeError:
            yhat = MODEL.predict(X)
    else:
        yhat = MODEL.predict(X)
    if TRANSFORM == "log1p":
        yhat = np.expm1(yhat)
    return yhat[0]

def get_top_features(X):
    """Get top 10 features by importance for display."""
    importances = MODEL.feature_importances_
    feat_imp = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])[:10]
    result = []
    for name, imp in feat_imp:
        val = float(X[name].iloc[0]) if name in X.columns else 0
        result.append({"name": name, "importance": round(float(imp), 4), "value": round(val, 6)})
    return result


if __name__ == "__main__":
    print("Starting FWI Chl-a Prediction Web App...")
    print("Open http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=False)
