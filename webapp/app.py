#!/usr/bin/env python3
"""
FWI Water Quality Prediction Web App
Flask backend serving the Chl-a and DO prediction API and frontend.
"""
import os, sys, json, math, asyncio, traceback
import numpy as np
import pandas as pd
import httpx
import ee
import joblib
import nest_asyncio
import google.auth
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime, timedelta
from pathlib import Path
from io import BytesIO

nest_asyncio.apply()

def sanitize(obj):
    """Replace NaN/Inf with None so JSON serialization works."""
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)          # ensure native Python float for JSON
    if isinstance(obj, (np.integer,)):
        return int(obj)            # ensure native Python int for JSON
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]
    return obj

app = Flask(__name__, static_folder="static")

# ── Paths ──
# BASE_DIR works both locally (parent of webapp/) and in Docker (/app)
BASE_DIR = Path(os.environ.get("APP_BASE_DIR", str(Path(__file__).resolve().parent.parent)))
CHLA_MODEL_PATH = str(BASE_DIR / "models" / "xgb_chla.joblib")
DO_V1_MODEL_PATH = str(BASE_DIR / "models" / "xgb_do_v1.joblib")
DO_V2_MODEL_PATH = str(BASE_DIR / "models" / "xgb_do.joblib")
ARA_KEY_PATH = str(BASE_DIR / "2026 Github ARA Pond IDs Key.csv")

# ── Load pond data at startup ──
PONDS_DF = pd.read_csv(ARA_KEY_PATH)
PONDS_WITH_GPS = PONDS_DF[PONDS_DF["latitude"].notna()].copy()

# ── Load Chl-a model at startup ──
CHLA_BUNDLE = joblib.load(CHLA_MODEL_PATH)
CHLA_MODEL = CHLA_BUNDLE["model"]
RAW_INPUTS = CHLA_BUNDLE["raw_inputs"]          # shared — both models use same inputs
CHLA_FEATURE_NAMES = CHLA_BUNDLE["feature_names"]
CHLA_TRANSFORM = CHLA_BUNDLE.get("transform", "none")
CHLA_BEST_ITER = CHLA_BUNDLE.get("best_iteration", None)

# ── Load DO Model 1 at startup ──
DO_V1_BUNDLE = joblib.load(DO_V1_MODEL_PATH)
DO_V1_MODEL = DO_V1_BUNDLE["model"]
DO_V1_FEATURE_NAMES = DO_V1_BUNDLE["feature_names"]
DO_V1_TRANSFORM = DO_V1_BUNDLE.get("transform", "none")
DO_V1_BEST_ITER = DO_V1_BUNDLE.get("best_iteration", None)

# ── Load DO Model 2 at startup ──
DO_V2_BUNDLE = joblib.load(DO_V2_MODEL_PATH)
DO_V2_MODEL = DO_V2_BUNDLE["model"]
DO_V2_FEATURE_NAMES = DO_V2_BUNDLE["feature_names"]
DO_V2_TRANSFORM = DO_V2_BUNDLE.get("transform", "none")
DO_V2_BEST_ITER = DO_V2_BUNDLE.get("best_iteration", None)

# Legacy aliases — DO_MODEL etc. point to Model 2 for backward compat
DO_MODEL = DO_V2_MODEL
DO_FEATURE_NAMES = DO_V2_FEATURE_NAMES
DO_TRANSFORM = DO_V2_TRANSFORM
DO_BEST_ITER = DO_V2_BEST_ITER

# Legacy aliases so batch code doesn't break
MODEL = CHLA_MODEL
FEATURE_NAMES = CHLA_FEATURE_NAMES
TRANSFORM = CHLA_TRANSFORM
BEST_ITERATION = CHLA_BEST_ITER

# ── GEE init ──
EE_INITIALIZED = False
GEE_PROJECT = os.environ.get("GEE_PROJECT", "ee-haven")

def init_ee():
    global EE_INITIALIZED
    if not EE_INITIALIZED:
        # On Cloud Run, use the default service account credentials automatically.
        # Locally, falls back to `earthengine authenticate` credentials.
        try:
            credentials, project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/earthengine"]
            )
            ee.Initialize(credentials=credentials, project=GEE_PROJECT or project)
            print(f"[GEE] Initialized with service account (project={GEE_PROJECT or project})")
        except Exception:
            # Fallback for local dev with `earthengine authenticate`
            ee.Initialize(project=GEE_PROJECT)
            print(f"[GEE] Initialized with default credentials (project={GEE_PROJECT})")
        EE_INITIALIZED = True

# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/example-csv")
def example_csv():
    """Serve an example batch CSV file."""
    from flask import Response
    csv_content = (
        "pond_id,lat,lon,2025-06-28,2025-07-13,2025-08-02\n"
        "NSR1,16.6627,81.7584,,,\n"
        "NRR1,16.8131,81.3153,,,\n"
        "AKR1,16.5698,80.9228,,,\n"
        ",13.0850,80.2700,,,\n"
    )
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=chla_batch_example.csv"},
    )

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

@app.route("/api/version")
def version():
    return jsonify({"version": "2026-02-27-v6-thumb-debug"})

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        date_str = data["date"]  # YYYY-MM-DD

        # Support either pond_id lookup OR direct lat/lon
        pond_id = data.get("pond_id")
        if pond_id:
            pond_row = PONDS_WITH_GPS[PONDS_WITH_GPS["internal_pond_id"] == pond_id]
            if pond_row.empty:
                return jsonify({"error": f"Pond {pond_id} not found or has no GPS"}), 400
            lat = float(pond_row.iloc[0]["latitude"])
            lon = float(pond_row.iloc[0]["longitude"])
        elif "lat" in data and "lon" in data:
            lat = float(data["lat"])
            lon = float(data["lon"])
            pond_id = f"GPS ({lat:.4f}, {lon:.4f})"
            pond_row = None
        else:
            return jsonify({"error": "Provide either pond_id or lat+lon"}), 400
        target_date = pd.Timestamp(date_str, tz="UTC")

        init_ee()

        # Step 1: Find most recent S2 image ON or BEFORE target date
        s2_result = find_recent_s2(lon, lat, target_date)
        if s2_result is None:
            return jsonify({"error": "No Sentinel-2 image found in the 30 days before this date"}), 404

        s2_data, s2_meta = s2_result

        # Step 2: Get weather for the S2 acquisition date
        s2_date = pd.Timestamp(s2_meta["acq_time"], tz="UTC")
        weather = get_weather(lat, lon, s2_date)

        # Step 3: Build feature rows and predict all models
        features_chla = build_feature_row(s2_data, s2_meta, weather, CHLA_FEATURE_NAMES)
        features_do_v1 = build_feature_row(s2_data, s2_meta, weather, DO_V1_FEATURE_NAMES)
        features_do_v2 = build_feature_row(s2_data, s2_meta, weather, DO_V2_FEATURE_NAMES)

        chla_pred = run_model(features_chla, CHLA_MODEL, CHLA_BEST_ITER, CHLA_TRANSFORM)
        do_v1_pred = run_model(features_do_v1, DO_V1_MODEL, DO_V1_BEST_ITER, DO_V1_TRANSFORM)
        do_v2_pred = run_model(features_do_v2, DO_V2_MODEL, DO_V2_BEST_ITER, DO_V2_TRANSFORM)

        return jsonify(sanitize({
            "prediction_chla": round(float(chla_pred)),
            "prediction_do_v1": round(float(do_v1_pred), 1),
            "prediction_do_v2": round(float(do_v2_pred), 1),
            "prediction_do": round(float(do_v2_pred), 1),  # backward compat
            "prediction": round(float(chla_pred)),  # backward compat
            "pond": {
                "id": pond_id,
                "lat": lat,
                "lon": lon,
                "region": str(pond_row.iloc[0]["region"]) if pond_row is not None else "",
                "village": str(pond_row.iloc[0]["village"]) if pond_row is not None and pd.notna(pond_row.iloc[0]["village"]) else "",
                "farmer": str(pond_row.iloc[0]["farmer_name"]) if pond_row is not None and pd.notna(pond_row.iloc[0]["farmer_name"]) else "",
            },
            "satellite": {
                "image_id": s2_meta.get("image_id", ""),
                "acq_time": s2_meta.get("acq_time", ""),
                "cloud_pct": s2_meta.get("cloud_pct"),
                "cp_probability": s2_meta.get("cp_probability"),
                "cp_source": s2_meta.get("cp_source", ""),
                "spacecraft": s2_meta.get("spacecraft", ""),
                "days_before_target": round((target_date - s2_date).total_seconds() / 86400, 1),
                "thumb_url": s2_meta.get("thumb_url"),
                "thumb_bounds": s2_meta.get("thumb_bounds"),
                "thumb_error": s2_meta.get("thumb_error"),
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
            "feature_importance_chla": get_top_features(features_chla, CHLA_MODEL, CHLA_FEATURE_NAMES),
            "feature_importance_do_v1": get_top_features(features_do_v1, DO_V1_MODEL, DO_V1_FEATURE_NAMES),
            "feature_importance_do_v2": get_top_features(features_do_v2, DO_V2_MODEL, DO_V2_FEATURE_NAMES),
            "feature_importance_do": get_top_features(features_do_v2, DO_V2_MODEL, DO_V2_FEATURE_NAMES),  # backward compat
            "feature_importance": get_top_features(features_chla, CHLA_MODEL, CHLA_FEATURE_NAMES),  # backward compat
        }))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# BATCH PREDICTION
# ══════════════════════════════════════════════════════════════

# In-memory store for batch job progress
_batch_jobs = {}

def _predict_one(pond_id, date_str, lat=None, lon=None):
    """Run a single prediction (Chl-a + DO). Accepts pond_id OR lat/lon. Returns dict with result or error."""
    label = pond_id or f"{lat},{lon}"
    try:
        # Resolve coordinates
        if lat is None or lon is None:
            pond_row = PONDS_WITH_GPS[PONDS_WITH_GPS["internal_pond_id"] == pond_id]
            if pond_row.empty:
                return {"pond_id": label, "date": date_str, "error": f"Pond not found or no GPS"}
            lat = float(pond_row.iloc[0]["latitude"])
            lon = float(pond_row.iloc[0]["longitude"])

        target_date = pd.Timestamp(date_str, tz="UTC")

        s2_result = find_recent_s2(lon, lat, target_date)
        if s2_result is None:
            return {"pond_id": label, "date": date_str, "error": "No S2 image in 30-day window"}

        s2_data, s2_meta = s2_result
        s2_date = pd.Timestamp(s2_meta["acq_time"], tz="UTC")
        weather = get_weather(lat, lon, s2_date)

        # Chl-a prediction
        features_chla = build_feature_row(s2_data, s2_meta, weather, CHLA_FEATURE_NAMES)
        chla_pred = run_model(features_chla, CHLA_MODEL, CHLA_BEST_ITER, CHLA_TRANSFORM)

        # DO predictions (both models)
        features_do_v1 = build_feature_row(s2_data, s2_meta, weather, DO_V1_FEATURE_NAMES)
        do_v1_pred = run_model(features_do_v1, DO_V1_MODEL, DO_V1_BEST_ITER, DO_V1_TRANSFORM)

        features_do_v2 = build_feature_row(s2_data, s2_meta, weather, DO_V2_FEATURE_NAMES)
        do_v2_pred = run_model(features_do_v2, DO_V2_MODEL, DO_V2_BEST_ITER, DO_V2_TRANSFORM)

        return {
            "pond_id": label,
            "date": date_str,
            "s2_date": s2_meta.get("acq_time", "").split(" ")[0],
            "cloud_pct": round(s2_meta.get("cloud_pct", 0), 1) if s2_meta.get("cloud_pct") is not None else None,
            "chla": round(float(chla_pred)),
            "do_v1": round(float(do_v1_pred), 1),
            "do_v2": round(float(do_v2_pred), 1),
            "do": round(float(do_v2_pred), 1),  # backward compat
            "error": None,
        }
    except Exception as e:
        return {"pond_id": label, "date": date_str, "error": str(e)}


@app.route("/api/batch", methods=["POST"])
def batch_predict():
    """
    Accepts {"requests": [{"pond_id": "...", "date": "YYYY-MM-DD"}, ...]}
    Returns a batch_id. Poll /api/batch/<batch_id> for progress.
    """
    import uuid, threading

    data = request.json or {}
    reqs = data.get("requests", [])
    if not reqs:
        return jsonify({"error": "No requests provided"}), 400
    if len(reqs) > 5000:
        return jsonify({"error": "Maximum 5000 predictions per batch"}), 400

    batch_id = str(uuid.uuid4())[:8]
    _batch_jobs[batch_id] = {
        "status": "running",
        "total": len(reqs),
        "completed": 0,
        "current_pond": "",
        "results": [],
    }

    def _run():
        init_ee()
        job = _batch_jobs[batch_id]
        for i, req in enumerate(reqs):
            if job["status"] == "cancelled":
                break
            pid = req.get("pond_id", "")
            dt = req.get("date", "")
            lat = req.get("lat")
            lon = req.get("lon")
            job["current_pond"] = pid or f"{lat},{lon}"
            result = _predict_one(pid, dt, lat=lat, lon=lon)
            job["results"].append(result)
            job["completed"] = i + 1
        if job["status"] != "cancelled":
            job["status"] = "done"
        job["current_pond"] = ""

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({"batch_id": batch_id, "total": len(reqs)})


@app.route("/api/batch/<batch_id>")
def batch_status(batch_id):
    """Poll for batch job progress."""
    job = _batch_jobs.get(batch_id)
    if not job:
        return jsonify({"error": "Batch not found"}), 404
    return jsonify(job)

@app.route("/api/batch/<batch_id>/cancel", methods=["POST"])
def batch_cancel(batch_id):
    """Stop a running batch job. Already-completed results are kept."""
    job = _batch_jobs.get(batch_id)
    if not job:
        return jsonify({"error": "Batch not found"}), 404
    if job["status"] == "running":
        job["status"] = "cancelled"
    return jsonify({"status": job["status"], "completed": job["completed"]})


@app.route("/api/parse-csv", methods=["POST"])
def parse_csv():
    """
    Upload a CSV/XLSX with columns: lat, lon, then one or more date columns.
    Each date column header should be a date (e.g. 2025-06-28).
    Returns validated requests as JSON.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    fname = f.filename.lower()

    try:
        if fname.endswith(".xlsx") or fname.endswith(".xls"):
            df = pd.read_excel(BytesIO(f.read()), dtype=str)
        else:
            df = pd.read_csv(BytesIO(f.read()), dtype=str)
    except Exception as e:
        return jsonify({"error": f"Could not parse file: {e}"}), 400

    # Normalize column names (strip whitespace, lowercase)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # --- Detect format ---
    # New format: lat, lon, then date columns
    # Legacy format: pond_id, date

    lat_col = None
    for c in ["lat", "latitude"]:
        if c in df.columns:
            lat_col = c
            break

    lon_col = None
    for c in ["lon", "long", "longitude", "lng"]:
        if c in df.columns:
            lon_col = c
            break

    # Also check for optional pond_id/label column
    label_col = None
    for c in ["pond_id", "pond", "label", "name", "id", "internal_pond_id"]:
        if c in df.columns:
            label_col = c
            break

    requests_out = []
    errors = []

    if lat_col and lon_col:
        # ── NEW FORMAT: lat, lon, date-columns ──
        # All columns that aren't lat/lon/label are treated as date columns
        skip_cols = {lat_col, lon_col}
        if label_col:
            skip_cols.add(label_col)
        date_cols = [c for c in df.columns if c not in skip_cols]

        if not date_cols:
            return jsonify({"error": "No date columns found. Add columns with date headers (e.g. 2025-06-28) after lat and lon."}), 400

        # Validate date column headers
        valid_date_cols = []
        for dc in date_cols:
            try:
                pd.to_datetime(dc)
                valid_date_cols.append(dc)
            except:
                errors.append(f"Column '{dc}' is not a recognized date — skipped")

        if not valid_date_cols:
            return jsonify({"error": "No valid date columns found. Column headers after lat/lon should be dates (e.g. 2025-06-28)."}), 400

        for i, row in df.iterrows():
            row_num = i + 2  # header is row 1
            # Parse lat/lon
            try:
                lat_val = float(row[lat_col])
                lon_val = float(row[lon_col])
            except (ValueError, TypeError):
                errors.append(f"Row {row_num}: invalid lat/lon '{row[lat_col]}', '{row[lon_col]}'")
                continue

            if not (-90 <= lat_val <= 90) or not (-180 <= lon_val <= 180):
                errors.append(f"Row {row_num}: lat/lon out of range ({lat_val}, {lon_val})")
                continue

            label = str(row[label_col]).strip() if label_col and pd.notna(row.get(label_col)) and str(row[label_col]).strip() else f"{lat_val:.4f}, {lon_val:.4f}"

            for dc in valid_date_cols:
                date_str = pd.to_datetime(dc).strftime("%Y-%m-%d")
                requests_out.append({
                    "pond_id": label,
                    "date": date_str,
                    "lat": lat_val,
                    "lon": lon_val,
                })

    elif label_col:
        # ── LEGACY FORMAT: pond_id, date ──
        date_col = None
        for candidate in ["date", "target_date", "prediction_date"]:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            return jsonify({"error": "Could not find a 'date' column. Expected columns: pond_id, date (or use lat, lon, date-columns format)"}), 400

        valid_ids = set(PONDS_WITH_GPS["internal_pond_id"].values)
        for i, row in df.iterrows():
            pid = str(row[label_col]).strip()
            raw_date = row[date_col]
            try:
                dt = pd.to_datetime(raw_date)
                date_str = dt.strftime("%Y-%m-%d")
            except:
                errors.append(f"Row {i+2}: invalid date '{raw_date}'")
                continue
            if pid not in valid_ids:
                errors.append(f"Row {i+2}: unknown pond_id '{pid}'")
                continue
            requests_out.append({"pond_id": pid, "date": date_str})
    else:
        return jsonify({"error": "Could not detect CSV format. Expected columns: lat, lon (plus date columns as headers) — or: pond_id, date"}), 400

    return jsonify({
        "requests": requests_out,
        "errors": errors,
        "total_valid": len(requests_out),
        "total_errors": len(errors),
    })


# ══════════════════════════════════════════════════════════════
# S2 EXTRACTION
# ══════════════════════════════════════════════════════════════

def find_recent_s2(lon, lat, target_date, max_days_back=30):
    """Find the most recent S2 L1C image ON or BEFORE target_date."""
    pt = ee.Geometry.Point(lon, lat)
    aoi = pt.buffer(100)

    # GEE filterDate end is EXCLUSIVE, so advance by 1 day to include target_date
    t_end = ee.Date(target_date.isoformat()).advance(1, "day")
    t_start = t_end.advance(-(max_days_back + 1), "day")

    col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
             .filterBounds(aoi)
             .filterDate(t_start, t_end)
             .sort("system:time_start", False))  # most recent first

    n = int(col.size().getInfo() or 0)
    if n == 0:
        return None

    s2_raw = ee.Image(col.first())       # unclipped — used for thumbnail
    s2 = s2_raw.clip(aoi)

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
        # Try nearest in time
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

    # Generate true-color thumbnail for map overlay
    try:
        thumb_region = pt.buffer(500)  # 500m around pond → ~1km × 1km
        thumb_geom = thumb_region.bounds()
        thumb_url = s2_raw.getThumbURL({
            "bands": ["B4", "B3", "B2"],
            "min": 0,
            "max": 3000,
            "dimensions": 512,
            "region": thumb_geom,
            "format": "png",
        })
        bounds_coords = thumb_geom.getInfo()["coordinates"][0]
        sw = [bounds_coords[0][1], bounds_coords[0][0]]
        ne = [bounds_coords[2][1], bounds_coords[2][0]]
        meta["thumb_url"] = thumb_url
        meta["thumb_bounds"] = [sw, ne]
    except Exception as e:
        import traceback
        traceback.print_exc()
        meta["thumb_error"] = str(e)

    return s2_data, meta


# ══════════════════════════════════════════════════════════════
# WEATHER
# ══════════════════════════════════════════════════════════════

COORD_PREC = 5
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

def build_feature_row(s2_data, s2_meta, weather, feature_names=None):
    """Build a one-row DataFrame aligned to a model's expected features.
    If feature_names is None, uses CHLA_FEATURE_NAMES (backward compat).
    """
    if feature_names is None:
        feature_names = CHLA_FEATURE_NAMES
    row = {}
    row.update(s2_data)
    row.update(weather)
    df = pd.DataFrame([row])
    X = pd.get_dummies(df.reindex(columns=RAW_INPUTS), drop_first=False)
    X = X.reindex(columns=feature_names, fill_value=0)
    return X

def run_model(X, model=None, best_iteration=None, transform=None):
    """Run prediction with a given model. Falls back to Chl-a globals for backward compat."""
    if model is None:
        model = CHLA_MODEL
    if best_iteration is None and model is CHLA_MODEL:
        best_iteration = CHLA_BEST_ITER
    if transform is None and model is CHLA_MODEL:
        transform = CHLA_TRANSFORM

    if best_iteration is not None:
        try:
            yhat = model.predict(X, iteration_range=(0, best_iteration + 1))
        except TypeError:
            yhat = model.predict(X)
    else:
        yhat = model.predict(X)
    if transform == "log1p":
        yhat = np.expm1(yhat)
    return yhat[0]

def get_top_features(X, model=None, feature_names=None):
    """Get top 10 features by importance for display."""
    if model is None:
        model = CHLA_MODEL
    if feature_names is None:
        feature_names = CHLA_FEATURE_NAMES
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]
    result = []
    for name, imp in feat_imp:
        val = float(X[name].iloc[0]) if name in X.columns else 0.0
        # Guard against NaN/Inf leaking into JSON
        if not math.isfinite(val):
            val = None
        else:
            val = round(val, 6)
        imp_val = float(imp)
        if not math.isfinite(imp_val):
            imp_val = 0.0
        else:
            imp_val = round(imp_val, 4)
        result.append({"name": name, "importance": imp_val, "value": val})
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
