#!/usr/bin/env python3
"""
Optimized bulk Chl-a prediction for all FWI ponds.

Groups ponds by geographic cluster (Nellore / Eluru) and extracts
Sentinel-2 band values for ALL ponds in a cluster from a single image
via ee.FeatureCollection.sampleRegions(), reducing GEE round trips
from ~7 per pond to ~4 per cluster.

Usage:
    python batch_all_ponds.py                        # all of 2025
    python batch_all_ponds.py --start 2025-01-01 --end 2025-03-31  # Q1 only
    python batch_all_ponds.py --dry-run               # show plan, no GEE calls
"""
import os, sys, argparse, asyncio, time, json
import numpy as np
import pandas as pd
import httpx
import ee
import joblib
import nest_asyncio

nest_asyncio.apply()

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "xgb_chla.joblib")
ARA_KEY_PATH = os.path.join(SCRIPT_DIR, "2026 Github ARA Pond IDs Key.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "batch_results")

# ── GEE ──
GEE_PROJECT = os.environ.get("GEE_PROJECT", "ee-haven")

# ── S2 Config ──
BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]
MAX_DAYS_BACK = 30
S2_COLLECTION = "COPERNICUS/S2_HARMONIZED"
CP_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"

# ── Weather Config (from predict_chla.py) ──
COORD_PREC = 5
CONCURRENCY = 12
RETRIES = 3
BASE_FORECAST = "https://api.open-meteo.com/v1/forecast"
BASE_HISTFOR = "https://historical-forecast-api.open-meteo.com/v1/forecast"
BASE_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
CUTOFF_HISTFC = pd.Timestamp(2022, 1, 1, tz="UTC")
RECENT_DAYS_BUF = 5
DAILY_VARS = "temperature_2m_mean,relative_humidity_2m_mean,cloud_cover_mean,wind_speed_10m_mean,precipitation_sum,shortwave_radiation_sum,sunshine_duration,weather_code"
HOURLY_VARS = "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m,shortwave_radiation,sunshine_duration"


# ═══════════════════════════════════════════════════════════
# GEE INITIALIZATION
# ═══════════════════════════════════════════════════════════

def init_ee():
    try:
        import google.auth
        credentials, project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/earthengine"]
        )
        ee.Initialize(credentials=credentials, project=GEE_PROJECT or project)
        print(f"[GEE] Initialized with service account (project={GEE_PROJECT or project})")
    except Exception:
        ee.Initialize(project=GEE_PROJECT)
        print(f"[GEE] Initialized with default credentials (project={GEE_PROJECT})")


# ═══════════════════════════════════════════════════════════
# CLUSTER-LEVEL S2 EXTRACTION
# ═══════════════════════════════════════════════════════════

def extract_s2_for_cluster(ponds_df, target_date):
    """
    Extract S2 band values for ALL ponds in a cluster using a mosaic of
    same-date images and buffer-median extraction (single GEE call for bands).

    Parameters
    ----------
    ponds_df : DataFrame with columns [internal_pond_id, latitude, longitude]
    target_date : pd.Timestamp (UTC)

    Returns
    -------
    list[dict] — one dict per pond with S2 data + metadata, or error
    """
    # Build bounding box from all pond coords, with buffer
    lats = ponds_df["latitude"].values
    lons = ponds_df["longitude"].values
    buf = 0.05  # ~5km buffer
    bbox = ee.Geometry.Rectangle([
        float(lons.min() - buf), float(lats.min() - buf),
        float(lons.max() + buf), float(lats.max() + buf),
    ])

    # GEE filterDate end is EXCLUSIVE, advance by 1 day to include target_date
    t_end = ee.Date(target_date.isoformat()).advance(1, "day")
    t_start = t_end.advance(-(MAX_DAYS_BACK + 1), "day")

    # Find most recent S2 images in window
    col = (ee.ImageCollection(S2_COLLECTION)
             .filterBounds(bbox)
             .filterDate(t_start, t_end)
             .sort("system:time_start", False))

    n = int(col.size().getInfo() or 0)
    if n == 0:
        return [{"pond_id": r["internal_pond_id"], "error": "No S2 image in 30-day window"}
                for _, r in ponds_df.iterrows()]

    # Get the most recent image to find its date
    first_img = ee.Image(col.first())

    # Extract image properties (shared for all ponds)
    prop_keys = [
        "system:index", "system:time_start", "CLOUDY_PIXEL_PERCENTAGE",
        "SPACECRAFT_NAME", "MEAN_SOLAR_AZIMUTH_ANGLE", "MEAN_SOLAR_ZENITH_ANGLE",
        "SOLAR_IRRADIANCE_B1", "MEAN_INCIDENCE_AZIMUTH_ANGLE_B1",
        "MEAN_INCIDENCE_ZENITH_ANGLE_B1", "REFLECTANCE_CONVERSION_CORRECTION",
    ]
    props = ee.Dictionary(first_img.toDictionary(prop_keys)).getInfo() or {}
    acq_ms = props.get("system:time_start")
    acq_time = ee.Date(acq_ms).format("YYYY-MM-dd HH:mm:ss").getInfo() if acq_ms else ""

    # ── Mosaic all same-date images (cluster may span multiple S2 tiles) ──
    acq_date = ee.Date(acq_ms)
    same_date_col = col.filter(ee.Filter.date(
        acq_date.format("YYYY-MM-dd"),
        acq_date.advance(1, "day"),
    ))
    mosaic = same_date_col.mosaic()

    # ── Cloud probability ──
    s2_idx = ee.String(first_img.get("system:index"))
    cp_all = ee.ImageCollection(CP_COLLECTION)
    cp_filt = cp_all.filter(ee.Filter.eq("system:index", s2_idx))
    cp_n = int(cp_filt.size().getInfo() or 0)

    cp_source = "none"
    cp_img = None
    if cp_n > 0:
        cp_img = ee.Image(cp_filt.first())
        cp_source = "index"
    else:
        # Try matching all same-date CP images and mosaic them
        cp_time_filt = (cp_all.filterBounds(bbox)
                        .filterDate(t_start, t_end)
                        .map(lambda c: ee.Image(c).set(
                            "td", ee.Number(c.get("system:time_start"))
                                      .subtract(ee.Number(acq_ms)).abs()))
                        .sort("td"))
        cp_time_n = int(cp_time_filt.size().getInfo() or 0)
        if cp_time_n > 0:
            cp_img = ee.Image(cp_time_filt.first())
            cp_source = "time"

    # For CP, mosaic same-date cloud probability images
    cp_mosaic = None
    if cp_source != "none":
        cp_same_date = (cp_all.filterBounds(bbox)
                        .filterDate(
                            acq_date.format("YYYY-MM-dd"),
                            acq_date.advance(1, "day")))
        if int(cp_same_date.size().getInfo() or 0) > 0:
            cp_mosaic = cp_same_date.mosaic()

    # ── Build image stack: scaled S2 bands ──
    scaled = mosaic.select(BANDS).divide(10000.0)

    # ── Create FeatureCollection of pond BUFFERS (15m ≈ 3×3 pixels at 10m) ──
    features = []
    for _, row in ponds_df.iterrows():
        pt = ee.Geometry.Point(float(row["longitude"]), float(row["latitude"]))
        buf_geom = pt.buffer(15)  # ~3×3 at 10m resolution
        features.append(ee.Feature(buf_geom, {"pond_id": row["internal_pond_id"]}))
    fc = ee.FeatureCollection(features)

    # Also create point-based FC for single-pixel CP extraction
    pt_features = []
    for _, row in ponds_df.iterrows():
        pt = ee.Geometry.Point(float(row["longitude"]), float(row["latitude"]))
        pt_features.append(ee.Feature(pt, {"pond_id": row["internal_pond_id"]}))
    fc_pts = ee.FeatureCollection(pt_features)

    # ── Extract median band values for ALL ponds in ONE call ──
    sampled = scaled.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.median(),
        scale=10,
    )
    sampled_list = sampled.getInfo()

    # ── Extract cloud probability median for each pond ──
    cp_at_pts = {}
    if cp_mosaic is not None:
        prob = cp_mosaic.select("probability")
        # Median CP within buffer
        cp_sampled = prob.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.median(),
            scale=10,
        )
        cp_info = cp_sampled.getInfo()
        for feat in cp_info.get("features", []):
            p = feat.get("properties", {})
            cp_at_pts[p.get("pond_id")] = p.get("probability")

        # Single-pixel CP at point center
        cp_pt_sampled = prob.reduceRegions(
            collection=fc_pts,
            reducer=ee.Reducer.first(),
            scale=10,
        )
        cp_pt_info = cp_pt_sampled.getInfo()
        for feat in cp_pt_info.get("features", []):
            p = feat.get("properties", {})
            # Override with point-level value (more precise)
            val = p.get("first") if p.get("first") is not None else p.get("probability")
            if val is not None:
                cp_at_pts[p.get("pond_id")] = val

    # ── Build result rows ──
    results = []
    sampled_by_id = {}
    for feat in sampled_list.get("features", []):
        p = feat.get("properties", {})
        sampled_by_id[p.get("pond_id")] = p

    for _, row in ponds_df.iterrows():
        pid = row["internal_pond_id"]
        p = sampled_by_id.get(pid)

        if p is None:
            results.append({"pond_id": pid, "error": "No S2 data at this location"})
            continue

        # Build s2_data dict matching app.py's format
        # reduceRegions with Reducer.median() returns band names as-is (e.g. "B3")
        # but the model expects "s2_med3_B3_median" format
        s2_data = {}
        for k, v in p.items():
            if k == "pond_id":
                continue
            s2_data[f"s2_med3_{k}_median"] = v

        # Check if any band has data
        if all(v is None for k, v in s2_data.items() if k.startswith("s2_med3_B")):
            results.append({"pond_id": pid, "error": "S2 bands all null (no coverage)"})
            continue

        # NDWI
        b3 = s2_data.get("s2_med3_B3_median")
        b8 = s2_data.get("s2_med3_B8_median")
        if b3 is not None and b8 is not None and (b3 + b8) != 0:
            s2_data["s2_med3_NDWI"] = (b3 - b8) / (b3 + b8)
        else:
            s2_data["s2_med3_NDWI"] = np.nan

        s2_data["s2_cloud_pct"] = props.get("CLOUDY_PIXEL_PERCENTAGE")
        s2_data["cp_probability"] = cp_at_pts.get(pid)
        s2_data["cp_source"] = cp_source
        s2_data["s2_med3_probability_median"] = cp_at_pts.get(pid)
        s2_data["s2_prop_MEAN_SOLAR_AZIMUTH_ANGLE"] = props.get("MEAN_SOLAR_AZIMUTH_ANGLE")
        s2_data["s2_prop_MEAN_SOLAR_ZENITH_ANGLE"] = props.get("MEAN_SOLAR_ZENITH_ANGLE")
        s2_data["s2_prop_SOLAR_IRRADIANCE_B1"] = props.get("SOLAR_IRRADIANCE_B1")
        s2_data["s2_prop_MEAN_INCIDENCE_AZIMUTH_ANGLE_B1"] = props.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_B1")
        s2_data["s2_prop_MEAN_INCIDENCE_ZENITH_ANGLE_B1"] = props.get("MEAN_INCIDENCE_ZENITH_ANGLE_B1")
        s2_data["RCC"] = props.get("REFLECTANCE_CONVERSION_CORRECTION")

        results.append({
            "pond_id": pid,
            "s2_data": s2_data,
            "image_id": props.get("system:index", ""),
            "acq_time": acq_time,
            "cloud_pct": props.get("CLOUDY_PIXEL_PERCENTAGE"),
            "spacecraft": props.get("SPACECRAFT_NAME", ""),
            "error": None,
        })

    return results


# ═══════════════════════════════════════════════════════════
# WEATHER (batched async, adapted from predict_chla.py)
# ═══════════════════════════════════════════════════════════

def _route_backend(sample_day_utc):
    now_utc = pd.Timestamp.now(tz="UTC")
    cutoff_recent = (now_utc - pd.Timedelta(days=RECENT_DAYS_BUF)).floor("D")
    if sample_day_utc > cutoff_recent:
        return BASE_FORECAST
    if sample_day_utc >= CUTOFF_HISTFC:
        return BASE_HISTFOR
    return BASE_ARCHIVE


async def _get(client, url, params):
    err = None
    for i in range(RETRIES):
        try:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            err = e
            await asyncio.sleep(1.5 * (i + 1))
    return {"error": str(err)}


def _as_float_arr(v):
    if v is None or len(v) == 0:
        return np.array([], dtype="float64")
    return pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(dtype="float64")


async def fetch_weather_batch(weather_requests):
    """
    Fetch weather for a list of (lat, lon, s2_date) tuples.
    Groups by location and backend, fetches with async concurrency.

    Returns dict: (lat, lon) → {date → {weather_vars}}
    """
    if not weather_requests:
        return {}

    # Group by rounded (lat, lon) and backend
    groups = {}  # (backend_name, lat_r, lon_r) → {start_date, end_date, s2_dates}
    for lat, lon, s2_date in weather_requests:
        day = s2_date.floor("D")
        backend = _route_backend(day)
        lat_r = round(lat, COORD_PREC)
        lon_r = round(lon, COORD_PREC)
        key = (backend, lat_r, lon_r)
        if key not in groups:
            groups[key] = {"lat": lat_r, "lon": lon_r, "days": set()}
        groups[key]["days"].add(day)

    # For each group, compute date range
    tasks_by_backend = {}
    for (backend, lat_r, lon_r), g in groups.items():
        min_day = min(g["days"])
        max_day = max(g["days"])
        start = (min_day - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        end_dminus1 = (max_day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end_d = max_day.strftime("%Y-%m-%d")
        tasks_by_backend.setdefault(backend, []).append({
            "lat": lat_r, "lon": lon_r,
            "start_date": start, "end_date_dminus1": end_dminus1,
            "end_date_d": end_d,
        })

    daily_results = {}  # (lat, lon) → DataFrame of rolled daily weather
    hourly_results = {}  # (lat, lon) → DataFrame of hourly weather

    limits = httpx.Limits(max_connections=CONCURRENCY*2, max_keepalive_connections=CONCURRENCY*2)
    timeout = httpx.Timeout(connect=10, read=60, write=10, pool=60)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        for backend_name, tasks in tasks_by_backend.items():
            base = {"forecast": BASE_FORECAST, "histfor": BASE_HISTFOR, "archive": BASE_ARCHIVE}[backend_name]

            # ── Daily (batch up to 60 locations per request) ──
            for chunk_start in range(0, len(tasks), 60):
                chunk = tasks[chunk_start:chunk_start+60]
                params = {
                    "latitude": ",".join(str(t["lat"]) for t in chunk),
                    "longitude": ",".join(str(t["lon"]) for t in chunk),
                    "daily": DAILY_VARS,
                    "start_date": min(t["start_date"] for t in chunk),
                    "end_date": max(t["end_date_dminus1"] for t in chunk),
                    "timezone": "UTC",
                    "cell_selection": "nearest",
                }
                js = await _get(client, base, params)
                if isinstance(js, dict) and js.get("error"):
                    print(f"  [Weather] Daily error: {js['error']}")
                    continue

                objs = js if isinstance(js, list) else [js]
                for o in objs:
                    lat_o = round(float(o.get("latitude")), COORD_PREC)
                    lon_o = round(float(o.get("longitude")), COORD_PREC)
                    d = o.get("daily", {}) or {}
                    dates = pd.to_datetime(d.get("time", []), utc=True).date
                    if len(dates) == 0:
                        continue
                    df_d = pd.DataFrame({
                        "date": dates,
                        "t2m_mean": _as_float_arr(d.get("temperature_2m_mean", [])),
                        "rh_mean": _as_float_arr(d.get("relative_humidity_2m_mean", [])),
                        "cloud_mean": _as_float_arr(d.get("cloud_cover_mean", [])),
                        "ws10_mean": _as_float_arr(d.get("wind_speed_10m_mean", [])),
                        "precip_sum": _as_float_arr(d.get("precipitation_sum", [])),
                        "swrad_sum": _as_float_arr(d.get("shortwave_radiation_sum", [])),
                        "sunshine_s": _as_float_arr(d.get("sunshine_duration", [])),
                        "weather_code": _as_float_arr(d.get("weather_code", [])),
                    }).set_index("date").sort_index()

                    roll_mean = df_d[["t2m_mean","rh_mean","cloud_mean","ws10_mean"]].rolling(3, min_periods=3).mean()
                    roll_sum = df_d[["precip_sum","swrad_sum","sunshine_s"]].rolling(3, min_periods=3).sum()
                    roll_wc = df_d[["weather_code"]].rolling(3, min_periods=3).apply(
                        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[-1], raw=False)
                    roll = pd.concat([roll_mean, roll_sum, roll_wc], axis=1)

                    # Map back to requested coords (Open-Meteo may snap)
                    best_key = None
                    best_d2 = 1e18
                    for t in chunk:
                        d2 = (t["lat"] - lat_o)**2 + (t["lon"] - lon_o)**2
                        if d2 < best_d2:
                            best_d2 = d2
                            best_key = (t["lat"], t["lon"])
                    if best_key:
                        daily_results[best_key] = roll

            # ── Hourly (one request per location) ──
            for t in tasks:
                params = {
                    "latitude": t["lat"], "longitude": t["lon"],
                    "hourly": HOURLY_VARS,
                    "start_date": t["start_date"], "end_date": t["end_date_d"],
                    "timezone": "UTC", "cell_selection": "nearest",
                }
                js = await _get(client, base, params)
                if isinstance(js, dict) and js.get("error"):
                    continue
                h = js.get("hourly", {}) or {}
                times = pd.to_datetime(h.get("time", []), utc=True)
                if len(times) == 0:
                    continue
                df_h = pd.DataFrame({"time": times})
                for k in ["temperature_2m","relative_humidity_2m","cloud_cover",
                          "wind_speed_10m","shortwave_radiation","precipitation","sunshine_duration"]:
                    df_h[k] = pd.to_numeric(pd.Series(h.get(k, [])), errors="coerce").values
                df_h = df_h.set_index("time").sort_index()
                hourly_results[(t["lat"], t["lon"])] = df_h

    return daily_results, hourly_results


def get_weather_for_row(lat, lon, s2_date, daily_results, hourly_results):
    """Look up pre-fetched weather for a single pond/date."""
    lat_r = round(lat, COORD_PREC)
    lon_r = round(lon, COORD_PREC)
    day = s2_date.floor("D")
    result = {}

    df_roll = daily_results.get((lat_r, lon_r))
    if df_roll is not None and not df_roll.empty:
        pick = (day - pd.Timedelta(days=1)).date()
        if pick in df_roll.index:
            v = df_roll.loc[pick]
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

    df_h = hourly_results.get((lat_r, lon_r))
    if df_h is not None and not df_h.empty:
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

    return result


# ═══════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════

def load_model():
    bundle = joblib.load(MODEL_PATH)
    return {
        "model": bundle["model"],
        "raw_inputs": bundle["raw_inputs"],
        "feature_names": bundle["feature_names"],
        "transform": bundle.get("transform", "none"),
        "best_iteration": bundle.get("best_iteration"),
    }


def predict_batch(rows, model_bundle):
    """
    Vectorized prediction for a batch of feature rows.

    Parameters
    ----------
    rows : list[dict] — each dict has s2_data + weather merged
    model_bundle : dict from load_model()

    Returns
    -------
    np.array of predicted Chl-a values
    """
    model = model_bundle["model"]
    raw_inputs = model_bundle["raw_inputs"]
    feature_names = model_bundle["feature_names"]
    transform = model_bundle["transform"]
    best_it = model_bundle["best_iteration"]

    df = pd.DataFrame(rows)
    X = pd.get_dummies(df.reindex(columns=raw_inputs), drop_first=False)
    X = X.reindex(columns=feature_names, fill_value=0)

    if best_it is not None:
        try:
            yhat = model.predict(X, iteration_range=(0, best_it + 1))
        except TypeError:
            yhat = model.predict(X)
    else:
        yhat = model.predict(X)

    if transform == "log1p":
        yhat = np.expm1(yhat)

    return yhat


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def generate_target_dates(start, end):
    """Generate target dates every 5 days (approximate S2 revisit)."""
    dates = pd.date_range(start, end, freq="5D", tz="UTC")
    return dates


def run_batch(start_date, end_date, dry_run=False):
    """Run optimized batch predictions for all ponds."""

    # ── Load data ──
    print(f"\n{'='*60}")
    print(f"FWI Bulk Chl-a Prediction")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}\n")

    ponds = pd.read_csv(ARA_KEY_PATH)
    ponds_gps = ponds[ponds["latitude"].notna()].copy()
    print(f"Loaded {len(ponds_gps)} ponds with GPS coordinates")

    clusters = {}
    for region, group in ponds_gps.groupby("region"):
        clusters[region] = group.reset_index(drop=True)
        print(f"  {region}: {len(group)} ponds")

    target_dates = generate_target_dates(start_date, end_date)
    total_predictions = len(ponds_gps) * len(target_dates)
    print(f"\nTarget dates: {len(target_dates)} (every 5 days)")
    print(f"Total predictions: {total_predictions}")

    if dry_run:
        print("\n[DRY RUN] Would process the above. Exiting.")
        return

    # ── Initialize ──
    init_ee()
    model_bundle = load_model()
    print(f"Model loaded: {len(model_bundle['feature_names'])} features, best_iteration={model_bundle['best_iteration']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    t_start_total = time.time()

    for date_idx, target_date in enumerate(target_dates):
        t_start_date = time.time()
        date_str = target_date.strftime("%Y-%m-%d")
        print(f"\n── Date {date_idx+1}/{len(target_dates)}: {date_str} ──")

        date_results = []

        # ── Step 1: Extract S2 for each cluster ──
        for region, cluster_ponds in clusters.items():
            print(f"  [{region}] Extracting S2 for {len(cluster_ponds)} ponds...", end=" ", flush=True)
            try:
                s2_results = extract_s2_for_cluster(cluster_ponds, target_date)
                n_ok = sum(1 for r in s2_results if r.get("error") is None)
                n_err = len(s2_results) - n_ok
                print(f"OK={n_ok}, errors={n_err}")
            except Exception as e:
                print(f"FAILED: {e}")
                s2_results = [{"pond_id": r["internal_pond_id"], "error": str(e)}
                              for _, r in cluster_ponds.iterrows()]

            for r in s2_results:
                r["region"] = region
                r["target_date"] = date_str
                # Attach lat/lon from ponds data
                pond_row = cluster_ponds[cluster_ponds["internal_pond_id"] == r["pond_id"]]
                if not pond_row.empty:
                    r["lat"] = float(pond_row.iloc[0]["latitude"])
                    r["lon"] = float(pond_row.iloc[0]["longitude"])
            date_results.extend(s2_results)

        # ── Step 2: Fetch weather for successful S2 results ──
        ok_results = [r for r in date_results if r.get("error") is None]
        if ok_results:
            weather_requests = []
            for r in ok_results:
                s2_date = pd.Timestamp(r["acq_time"], tz="UTC")
                weather_requests.append((r["lat"], r["lon"], s2_date))

            print(f"  Fetching weather for {len(weather_requests)} ponds...", end=" ", flush=True)
            try:
                daily_results, hourly_results = asyncio.run(
                    fetch_weather_batch(weather_requests)
                )
                print("done")
            except Exception as e:
                print(f"FAILED: {e}")
                daily_results, hourly_results = {}, {}

            # ── Step 3: Build features and predict ──
            feature_rows = []
            feature_indices = []  # track which ok_results index maps to which feature row
            for i, r in enumerate(ok_results):
                s2_date = pd.Timestamp(r["acq_time"], tz="UTC")
                weather = get_weather_for_row(r["lat"], r["lon"], s2_date, daily_results, hourly_results)

                row = {}
                row.update(r["s2_data"])
                row.update(weather)
                feature_rows.append(row)
                feature_indices.append(i)

            if feature_rows:
                predictions = predict_batch(feature_rows, model_bundle)
                for idx, pred in zip(feature_indices, predictions):
                    ok_results[idx]["chla"] = round(float(pred))

        # ── Collect results ──
        for r in date_results:
            out_row = {
                "pond_id": r.get("pond_id"),
                "region": r.get("region"),
                "lat": r.get("lat"),
                "lon": r.get("lon"),
                "target_date": r.get("target_date"),
                "s2_date": r.get("acq_time", "").split(" ")[0] if r.get("acq_time") else "",
                "s2_image_id": r.get("image_id", ""),
                "cloud_pct": r.get("cloud_pct"),
                "chla": r.get("chla"),
                "error": r.get("error"),
            }
            all_results.append(out_row)

        elapsed = time.time() - t_start_date
        n_ok = sum(1 for r in date_results if r.get("chla") is not None)
        print(f"  → {n_ok}/{len(date_results)} predictions in {elapsed:.1f}s")

        # Save intermediate results every 10 dates
        if (date_idx + 1) % 10 == 0:
            _save_results(all_results, "intermediate")

    # ── Final save ──
    total_elapsed = time.time() - t_start_total
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(all_results)} rows in {total_elapsed/60:.1f} minutes")
    n_ok = sum(1 for r in all_results if r.get("chla") is not None)
    n_err = sum(1 for r in all_results if r.get("error") is not None)
    print(f"  Predictions: {n_ok}, Errors: {n_err}")
    _save_results(all_results, f"chla_{start_date}_to_{end_date}")

    return all_results


def _save_results(results, label):
    df = pd.DataFrame(results)
    path = os.path.join(OUTPUT_DIR, f"{label}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path} ({len(df)} rows)")


# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk Chl-a prediction for all FWI ponds")
    parser.add_argument("--start", default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running")
    args = parser.parse_args()

    run_batch(args.start, args.end, dry_run=args.dry_run)
