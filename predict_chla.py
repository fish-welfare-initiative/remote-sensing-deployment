#!/usr/bin/env python3
"""
Self-contained Python script for predicting chlorophyll-a (Chl-a) using:
  1. Sentinel-2 L1C data from Google Earth Engine
  2. Weather data from Open-Meteo
  3. XGBoost Chl-a model

Usage:
    from predict_chla import predict_chla

    sites = [
        {"lat": 10.5, "lon": 75.5, "datetime": "2023-06-15 10:30:00"},
        {"lat": 10.6, "lon": 75.6, "datetime": "2023-06-16 11:00:00"},
    ]

    result_df = predict_chla(sites)
"""

import os
import sys
import asyncio
import numpy as np
import pandas as pd
import httpx
import nest_asyncio
from datetime import datetime
from typing import List, Optional, Tuple, Dict
import ee
import joblib

# Apply async compatibility patch
nest_asyncio.apply()

# ============================================================================
# GOOGLE EARTH ENGINE INITIALIZATION
# ============================================================================

def initialize_ee():
    """Initialize Google Earth Engine with saved credentials."""
    try:
        ee.Initialize(project='')
        print("[GEE] Initialized successfully")
    except Exception as e:
        print(f"[GEE] Init error: {e}")
        raise


# ============================================================================
# HELPER FUNCTIONS FOR EARTH ENGINE
# ============================================================================

def _prefix_props(props: dict, prefix: str = 's2_prop_') -> dict:
    """Prefix keys in a dictionary with a fixed string."""
    return {f'{prefix}{k}': v for k, v in (props or {}).items()}


def _extract_props_subset(img: ee.Image, keys: List[str]) -> dict:
    """Safely extract a subset of image properties in one round trip."""
    try:
        d = ee.Dictionary(img.toDictionary(keys)).getInfo() or {}
        return {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in d.items()}
    except Exception:
        return {}


def s2_attach_cloudprob_clipped(
    s2_img: ee.Image,
    cp_col: ee.ImageCollection,
    *,
    aoi_clip: Optional[ee.Geometry] = None,
    max_diff_ms: int = 24*60*60*1000
) -> ee.Image:
    """Attach the S2Cloudless probability band to a Sentinel-2 image."""
    idx = ee.String(s2_img.get('system:index'))
    t0  = ee.Number(s2_img.get('system:time_start'))

    # Try exact index first
    cp_idx  = cp_col.filter(ee.Filter.eq('system:index', idx)).first()
    # Else nearest-in-time within window
    cp_time = cp_col.map(lambda c: ee.Image(c).set(
                    'td', ee.Number(c.get('system:time_start')).subtract(t0).abs())
              ).filter(ee.Filter.lte('td', max_diff_ms)).sort('td').first()

    cp_use = ee.Image(ee.Algorithms.If(cp_idx, cp_idx,
                     ee.Algorithms.If(cp_time, cp_time, None)))

    prob = ee.Image(ee.Algorithms.If(
        cp_use, ee.Image(cp_use).select('probability'), ee.Image().select()
    ))
    if aoi_clip:
        prob = ee.Image(ee.Algorithms.If(
            prob.bandNames().size(), ee.Image(prob).clip(aoi_clip), prob
        ))

    cp_src = ee.String(ee.Algorithms.If(cp_idx, 'index',
                 ee.Algorithms.If(cp_time, 'time', 'none')))
    cp_td  = ee.Number(ee.Algorithms.If(cp_time, ee.Number(cp_time.get('td')), None))

    return s2_img.addBands(prob.rename(['probability'])) \
                 .set({'cp_source': cp_src, 'cp_time_diff_ms': cp_td})


def find_closest_s2_and_clip(
    lon: float, lat: float, when,
    *, days_window=1,
    s2_collection='COPERNICUS/S2_HARMONIZED',
    cloud_pct_max=None,
    clip_radius_m=100,
    verbose=False
):
    """Find the closest Sentinel-2 image to a sample and clip it."""
    t0 = ee.Date(when)
    pt = ee.Geometry.Point(lon, lat)
    aoi = pt.buffer(clip_radius_m)

    col = (ee.ImageCollection(s2_collection)
             .filterBounds(aoi)
             .filterDate(t0.advance(-days_window, 'day'), t0.advance(days_window, 'day')))
    if cloud_pct_max is not None:
        col = col.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_pct_max))

    col = col.map(lambda im: ee.Image(im).set(
        'td', ee.Number(im.get('system:time_start')).subtract(t0.millis()).abs()
    ))

    # Safe: ask size first; avoid .first() on empty
    try:
        n = int(col.size().getInfo() or 0)
    except Exception:
        n = 0

    meta = {'candidates': n}
    if n == 0:
        return None, aoi, meta

    # Pick closest in time
    col = col.sort('td')
    s2 = ee.Image(col.first()).clip(aoi)
    return s2, aoi, meta


def process_fast_clipped(
    in_df: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    time_col: str,
    days_window: float = 0.125,  # 3 hours = 0.125 days
    cloud_pct_max: Optional[float] = None,
    s2_clip_radius_m: int = 100,
    cp_clip_radius_m: int = 100,
    neighborhood_px: int = 1,
    scale_m: int = 10,
    bands_to_sample: Optional[List[str]] = None,
    verbose: bool = False,
    print_every: int = 50
) -> pd.DataFrame:
    """
    Process each row: extract Sentinel-2 bands, cloud probability, and image properties.
    Returns DataFrame with S2 features + original columns.
    """
    if bands_to_sample is None:
        bands_to_sample = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']

    cp_all = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    out = []

    for i, row in in_df.reset_index(drop=True).iterrows():
        try:
            lon, lat = float(row[lon_col]), float(row[lat_col])
        except Exception:
            if verbose:
                print(f"[row {i}] bad lat/lon")
            continue

        t = pd.to_datetime(row.get(time_col))
        if pd.isna(t):
            if verbose:
                print(f"[row {i}] missing time")
            continue

        # Make naive UTC if tz-aware
        if t.tzinfo is not None:
            t = t.tz_convert('UTC').tz_localize(None)

        # 1) Find nearest S2 in ±days_window
        s2_clip, s2_aoi, meta = find_closest_s2_and_clip(
            lon, lat, t, days_window=days_window,
            s2_collection='COPERNICUS/S2_HARMONIZED',
            cloud_pct_max=cloud_pct_max,
            clip_radius_m=s2_clip_radius_m,
            verbose=verbose
        )
        if s2_clip is None:
            base = dict(row)
            base.update({
                's2_found': False,
                'error': 'no_image_in_window',
                'cand': meta.get('candidates', 0)
            })
            out.append(base)
            if verbose and (i % print_every == 0):
                print(f"[row {i}] no S2 in window (±{days_window}d)")
            continue

        # 2) Attach cloud probability within AOI/time window
        cp_filt = (cp_all
                   .filterBounds(s2_aoi.buffer(cp_clip_radius_m))
                   .filterDate(ee.Date(t).advance(-days_window, 'day'),
                               ee.Date(t).advance(days_window, 'day')))
        try:
            s2cp = s2_attach_cloudprob_clipped(s2_clip, cp_filt, aoi_clip=s2_aoi,
                                               max_diff_ms=int(days_window*24*60*60*1000))
        except Exception:
            base = dict(row)
            base.update({'s2_found': False, 'error': 'cp_attach_failed'})
            out.append(base)
            if verbose and (i % print_every == 0):
                print(f"[row {i}] CP attach failed")
            continue

        # 3) 3x3 medians for requested bands
        scaled = s2cp.select(bands_to_sample).divide(10000.0)  # L1C scaled to 1e4

        # DON'T rename bands before reduceNeighborhood — it already appends _median
        prob_orig = s2cp.select('probability')                  # 0..100 UINT8
        work = ee.Image.cat([scaled, prob_orig])

        med3 = work.reduceNeighborhood(
            reducer=ee.Reducer.median(),
            kernel=ee.Kernel.square(neighborhood_px)
        )

        pt = ee.Geometry.Point(lon, lat)
        try:
            vals = med3.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=pt, scale=scale_m, bestEffort=True, maxPixels=1e13
            ).getInfo() or {}
        except Exception:
            base = dict(row)
            base.update({'s2_found': False, 'error': 'reduceRegion_failed'})
            out.append(base)
            if verbose and (i % print_every == 0):
                print(f"[row {i}] reduceRegion failed")
            continue

        # Single-pixel CP at the point (from original non-reduced image)
        try:
            cp_at_pt = prob_orig.reduceRegion(
                ee.Reducer.first(), pt, scale=scale_m, bestEffort=True, maxPixels=1e13
            ).getInfo()
            cp_at_pt = list(cp_at_pt.values())[0] if isinstance(cp_at_pt, dict) and cp_at_pt else None
        except Exception:
            cp_at_pt = None

        # Extract image properties
        keep_keys = [
            'system:index', 'system:time_start',
            'CLOUDY_PIXEL_PERCENTAGE', 'CLOUD_COVERAGE_ASSESSMENT',
            'DATATAKE_IDENTIFIER', 'PRODUCT_ID', 'SPACECRAFT_NAME',
            'MEAN_SOLAR_AZIMUTH_ANGLE', 'MEAN_SOLAR_ZENITH_ANGLE',
            'SOLAR_IRRADIANCE_B1', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B1',
            'MEAN_INCIDENCE_ZENITH_ANGLE_B1', 'REFLECTANCE_CONVERSION_CORRECTION',
            'cp_source', 'cp_time_diff_ms', 'td'
        ]
        props = _extract_props_subset(s2cp, keep_keys)

        # Compute time difference (seconds)
        try:
            td_s = float(props.get('td')) / 1000.0 if props.get('td') is not None else None
        except Exception:
            td_s = None

        base = dict(row)
        base.update({
            's2_found': True,
            's2_image_id': props.get('system:index'),
            's2_acq_time': ee.Date(props.get('system:time_start')).format('YYYY-MM-dd HH:mm:ss').getInfo()
                           if props.get('system:time_start') is not None else None,
            's2_cloud_pct': props.get('CLOUDY_PIXEL_PERCENTAGE') or props.get('CLOUD_COVERAGE_ASSESSMENT'),
            's2_time_diff_seconds': td_s,
            'cp_probability': cp_at_pt,
            'cp_time_diff_ms': props.get('cp_time_diff_ms'),
            'cp_source': props.get('cp_source'),
            's2_prop_PRODUCT_ID': props.get('PRODUCT_ID'),
            's2_prop_MEAN_SOLAR_AZIMUTH_ANGLE': props.get('MEAN_SOLAR_AZIMUTH_ANGLE'),
            's2_prop_MEAN_SOLAR_ZENITH_ANGLE': props.get('MEAN_SOLAR_ZENITH_ANGLE'),
            's2_prop_SOLAR_IRRADIANCE_B1': props.get('SOLAR_IRRADIANCE_B1'),
            's2_prop_MEAN_INCIDENCE_AZIMUTH_ANGLE_B1': props.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_B1'),
            's2_prop_MEAN_INCIDENCE_ZENITH_ANGLE_B1': props.get('MEAN_INCIDENCE_ZENITH_ANGLE_B1'),
            'RCC': props.get('REFLECTANCE_CONVERSION_CORRECTION'),
        })

        # Add 3x3 medians for each band + probability
        # reduceNeighborhood appends '_median' to each band name: B3 -> B3_median
        for k, v in vals.items():
            base[f's2_med3_{k}'] = v

        # Compute NDWI from medians if available
        b3m = base.get('s2_med3_B3_median')
        b8m = base.get('s2_med3_B8_median')
        if (b3m is not None) and (b8m is not None) and (b3m + b8m) != 0:
            base['s2_med3_NDWI'] = (b3m - b8m) / (b3m + b8m)
        else:
            base['s2_med3_NDWI'] = np.nan

        out.append(base)

        if verbose and (i % print_every == 0):
            print(f"[row {i}] ok  cand={meta.get('candidates', 'NA')}  id={props.get('system:index')}  cp@pt={cp_at_pt}")

    return pd.DataFrame(out)


# ============================================================================
# OPEN-METEO WEATHER FUNCTIONS
# ============================================================================

# Config constants
TZ_DAILY = "UTC"
COORD_PREC = 5
CONCURRENCY = 12
RETRIES = 3
CELL_SELECT = "nearest"
INCLUDE_HOURLY = True

BASE_FORECAST = "https://api.open-meteo.com/v1/forecast"
BASE_HISTFOR = "https://historical-forecast-api.open-meteo.com/v1/forecast"
BASE_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

RECENT_DAYS_BUF = 5
CUTOFF_HISTFC = pd.Timestamp(2022, 1, 1, tz="UTC")

DAILY_VARS = ",".join([
    "temperature_2m_mean", "relative_humidity_2m_mean", "cloud_cover_mean",
    "wind_speed_10m_mean", "precipitation_sum", "shortwave_radiation_sum",
    "sunshine_duration", "weather_code"
])
HOURLY_VARS = ",".join([
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "cloud_cover", "wind_speed_10m", "shortwave_radiation", "sunshine_duration"
])


def _as_float_arr(v):
    """Convert to float array safely."""
    if v is None or len(v) == 0:
        return np.array([], dtype="float64")
    return pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(dtype="float64")


def _batches(items, n):
    """Batch iterator."""
    buf = []
    for it in items:
        buf.append(it)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


async def _get(client, url, params):
    """Async HTTP GET with retries."""
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


def _route_backend(sample_day_utc, now_utc=None):
    """Route to correct Open-Meteo backend based on date."""
    if now_utc is None:
        now_utc = pd.Timestamp.now(tz="UTC")
    cutoff_recent = (now_utc - pd.Timedelta(days=RECENT_DAYS_BUF)).floor("D")
    if sample_day_utc > cutoff_recent:
        return "forecast"
    if sample_day_utc >= CUTOFF_HISTFC:
        return "histfor"
    return "archive"


def _rekey_to_requests(ret_map, tasks):
    """Map returned coords to requested coords by nearest distance."""
    if not ret_map:
        return {}
    ret_keys = list(ret_map.keys())
    out = {}
    for t in tasks:
        la, lo = float(t["lat"]), float(t["lon"])
        best_k, best_d2 = None, 1e18
        for (rla, rlo) in ret_keys:
            d2 = (rla - la) * (rla - la) + (rlo - lo) * (rlo - lo)
            if d2 < best_d2:
                best_d2 = d2
                best_k = (rla, rlo)
        out[(la, lo)] = ret_map[best_k]
    return out


def _prep_and_group(df_in, lat="lat", lon="lon", dtcol="datetime"):
    """Prepare and group data by backend and location."""
    df = df_in.reset_index(drop=True).copy()

    dt = pd.to_datetime(df[dtcol], utc=True, errors="coerce")
    df["_has_dt"] = dt.notna()
    df["_end_hour"] = dt.dt.floor("h")
    df["_day"] = dt.dt.floor("D")

    # Daily window [D-3..D-1]
    df["_start_day"] = (df["_day"] - pd.Timedelta(days=3)).dt.strftime("%Y-%m-%d")
    df["_end_day"] = (df["_day"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")

    df["_backend"] = np.where(df["_has_dt"], df["_day"].apply(_route_backend), None)

    groups = []
    for bname in ["forecast", "histfor", "archive"]:
        sub = df[(df["_backend"] == bname) & df["_has_dt"]].copy()
        if sub.empty:
            groups.append((bname, []))
            continue
        sub["LAT_R"] = sub[lat].round(COORD_PREC)
        sub["LON_R"] = sub[lon].round(COORD_PREC)
        out = []
        for (la, lo), g in sub.groupby(["LAT_R", "LON_R"]):
            min_start = (g["_day"].min() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            max_end_dminus1 = (g["_day"].max() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            max_end_d = (g["_day"].max()).strftime("%Y-%m-%d")
            out.append({
                "lat": float(la), "lon": float(lo),
                "start_date": min_start, "end_date_dminus1": max_end_dminus1,
                "end_date_d": max_end_d,
                "rows_index": g.index,
            })
        groups.append((bname, out))
    return df, dict(groups)


async def _fetch_daily_range(client, base, tasks):
    """Fetch daily weather data from Open-Meteo."""
    result = {}
    for chunk in list(_batches(tasks, 60)):
        by_win = {}
        for t in chunk:
            by_win.setdefault((t["start_date"], t["end_date_dminus1"]), []).append(t)
        for (start, end), sub in by_win.items():
            params = dict(
                latitude=",".join(str(t["lat"]) for t in sub),
                longitude=",".join(str(t["lon"]) for t in sub),
                daily=DAILY_VARS, start_date=start, end_date=end,
                timezone=TZ_DAILY, cell_selection=CELL_SELECT
            )
            js = await _get(client, base, params)
            if isinstance(js, dict) and js.get("error"):
                for t in sub:
                    result[(t["lat"], t["lon"])] = {"_error": js["error"]}
                continue
            objs = js if isinstance(js, list) else [js]
            ret_map = {}
            for o in objs:
                lat_o = float(o.get("latitude"))
                lon_o = float(o.get("longitude"))
                d = o.get("daily", {}) or {}
                dates = pd.to_datetime(d.get("time", []), utc=True).date
                if len(dates) == 0:
                    ret_map[(lat_o, lon_o)] = pd.DataFrame(index=pd.Index([], name="date"))
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

                roll_mean = df_d[["t2m_mean", "rh_mean", "cloud_mean", "ws10_mean"]].rolling(3, min_periods=3).mean()
                roll_sum = df_d[["precip_sum", "swrad_sum", "sunshine_s"]].rolling(3, min_periods=3).sum()
                roll_wc = df_d[["weather_code"]].rolling(3, min_periods=3).apply(
                    lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[-1], raw=False
                )

                ret_map[(lat_o, lon_o)] = pd.concat([roll_mean, roll_sum, roll_wc], axis=1)

            result.update(_rekey_to_requests(ret_map, sub))
    return result


async def _fetch_hourly_range(client, base, tasks):
    """Fetch hourly weather data from Open-Meteo."""
    result = {}
    for t in tasks:
        params = dict(
            latitude=t["lat"], longitude=t["lon"], hourly=HOURLY_VARS,
            start_date=t["start_date"], end_date=t["end_date_d"],
            timezone="UTC", cell_selection=CELL_SELECT
        )
        js = await _get(client, base, params)
        if isinstance(js, dict) and js.get("error"):
            result[(t["lat"], t["lon"])] = {"_error": js["error"]}
            continue
        h = js.get("hourly", {}) or {}
        times = pd.to_datetime(h.get("time", []), utc=True)
        if times.empty:
            df_h = pd.DataFrame(index=pd.DatetimeIndex([], name="time", tz="UTC"))
        else:
            df_h = pd.DataFrame({"time": times})
            for k in ["temperature_2m", "relative_humidity_2m", "cloud_cover",
                      "wind_speed_10m", "shortwave_radiation", "precipitation", "sunshine_duration"]:
                df_h[k] = pd.to_numeric(pd.Series(h.get(k, [])), errors="coerce").values
            df_h = df_h.set_index("time").sort_index()
        result[(t["lat"], t["lon"])] = df_h
    return result


async def openmeteo_range_join(df_in, lat="lat", lon="lon", dtcol="datetime", include_hourly=INCLUDE_HOURLY):
    """
    Fetch and join Open-Meteo weather data to input DataFrame.
    Adds daily and hourly weather columns.
    """
    base_df, groups = _prep_and_group(df_in, lat=lat, lon=lon, dtcol=dtcol)

    out = df_in.copy().reset_index(drop=True)

    assert len(out) == len(base_df), "Index mismatch"
    assert isinstance(out.index, pd.RangeIndex) and out.index.start == 0 and out.index.step == 1, \
        "Output index must be RangeIndex 0..n-1"

    limits = httpx.Limits(max_connections=CONCURRENCY*2, max_keepalive_connections=CONCURRENCY*2)
    timeout = httpx.Timeout(connect=10, read=40, write=10, pool=40)

    daily_idx, hourly_idx = {}, {}
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        for name, tasks in groups.items():
            if not tasks:
                continue
            base = {"forecast": BASE_FORECAST, "histfor": BASE_HISTFOR, "archive": BASE_ARCHIVE}[name]
            d = await _fetch_daily_range(client, base, tasks)
            daily_idx.update({(name, *k): v for k, v in d.items()})
            if include_hourly:
                h = await _fetch_hourly_range(client, base, tasks)
                hourly_idx.update({(name, *k): v for k, v in h.items()})

    # Initialize weather columns
    daily_cols = ["t2m_3d", "rh_3d", "cloud_3d", "ws10_3d", "precip_3d_sum",
                  "swrad_3d_sum", "sunshine_3d_sum_s", "weather_code_3d_mode",
                  "t2m_3d_mean", "rh_3d_mean", "cloud_3d_mean", "ws10_3d_mean"]
    hourly_cols = ["t2m_inst", "rh_inst", "cloud_inst", "ws10_inst", "swrad_inst",
                   "precip_inst", "sunshine_inst_s"]

    for c in daily_cols + hourly_cols:
        out[c] = np.nan

    # Fill in weather data
    for name, tasks in groups.items():
        if not tasks:
            continue
        for t in tasks:
            key = (name, float(t["lat"]), float(t["lon"]))
            df_roll = daily_idx.get(key, None)
            df_hour = hourly_idx.get(key, None) if include_hourly else None

            for i in t["rows_index"]:
                j = int(i)
                if j < 0 or j >= len(out):
                    continue

                r = base_df.loc[j]

                # DAILY: pick D-1 (representing [D-3..D-1] aggregate)
                if isinstance(df_roll, pd.DataFrame) and not df_roll.empty and pd.notna(r["_day"]):
                    day_pick = (r["_day"] - pd.Timedelta(days=1)).date()
                    if day_pick in df_roll.index:
                        val = df_roll.loc[day_pick]
                        out.loc[j, ["t2m_3d", "rh_3d", "cloud_3d", "ws10_3d",
                                    "precip_3d_sum", "swrad_3d_sum", "sunshine_3d_sum_s"]] = [
                            float(val.get("t2m_mean", np.nan)),
                            float(val.get("rh_mean", np.nan)),
                            float(val.get("cloud_mean", np.nan)),
                            float(val.get("ws10_mean", np.nan)),
                            float(val.get("precip_sum", np.nan)),
                            float(val.get("swrad_sum", np.nan)),
                            float(val.get("sunshine_s", np.nan)),
                        ]
                        # Weather code mode
                        wc = val.get("weather_code", np.nan)
                        if pd.notna(wc):
                            out.loc[j, "weather_code_3d_mode"] = float(wc)

                        # Compute 3-day means for these variables
                        out.loc[j, "t2m_3d_mean"] = float(val.get("t2m_mean", np.nan))
                        out.loc[j, "rh_3d_mean"] = float(val.get("rh_mean", np.nan))
                        out.loc[j, "cloud_3d_mean"] = float(val.get("cloud_mean", np.nan))
                        out.loc[j, "ws10_3d_mean"] = float(val.get("ws10_mean", np.nan))

                # HOURLY: exact hour or nearest within ±1h
                if include_hourly and isinstance(df_hour, pd.DataFrame) and not df_hour.empty and pd.notna(r["_end_hour"]):
                    t0 = r["_end_hour"]
                    idxr = df_hour.index.get_indexer([t0], method="nearest", tolerance=pd.Timedelta(hours=1))
                    if idxr.size and idxr[0] != -1:
                        hv = df_hour.iloc[idxr[0]]
                        out.loc[j, ["t2m_inst", "rh_inst", "cloud_inst", "ws10_inst",
                                    "swrad_inst", "precip_inst", "sunshine_inst_s"]] = [
                            float(hv.get("temperature_2m", np.nan)),
                            float(hv.get("relative_humidity_2m", np.nan)),
                            float(hv.get("cloud_cover", np.nan)),
                            float(hv.get("wind_speed_10m", np.nan)),
                            float(hv.get("shortwave_radiation", np.nan)),
                            float(hv.get("precipitation", np.nan)),
                            float(hv.get("sunshine_duration", np.nan)),
                        ]

    n = len(out)
    filled_daily = out[daily_cols].notna().any(axis=1).sum()
    filled_hour = out[hourly_cols].notna().any(axis=1).sum()
    print(f"[Open-Meteo] daily {filled_daily}/{n} | hourly {filled_hour}/{n}")

    return out


# ============================================================================
# MODEL PREDICTION
# ============================================================================

def predict_chla(sites: List[dict], model_path: str = "/sessions/dazzling-zealous-ritchie/mnt/RS.v4/Sol code/models/xgb_chla.joblib") -> pd.DataFrame:
    """
    Predict chlorophyll-a for a list of sites.

    Parameters
    ----------
    sites : list[dict]
        List of dictionaries with keys: "lat" (float), "lon" (float), "datetime" (str or datetime).
        datetime format: "YYYY-MM-DD HH:MM:SS" (UTC).

    model_path : str
        Path to the XGBoost Chl-a model joblib file.

    Returns
    -------
    pd.DataFrame
        DataFrame with input columns + all features + "pred_Chla" column.
    """
    # Initialize EE
    initialize_ee()

    # Convert input to DataFrame
    sites_df = pd.DataFrame(sites)
    sites_df['datetime'] = pd.to_datetime(sites_df['datetime'], utc=True)

    print(f"\n[predict_chla] Processing {len(sites_df)} sites...")

    # Step 1: Fetch weather data
    print("\n[1/3] Fetching Open-Meteo weather data...")
    df_weather = asyncio.run(openmeteo_range_join(
        sites_df, lat="lat", lon="lon", dtcol="datetime", include_hourly=True
    ))

    # Step 2: Fetch Sentinel-2 data
    print("\n[2/3] Fetching Sentinel-2 and cloud probability data...")
    df_s2 = process_fast_clipped(
        df_weather,
        lon_col="lon", lat_col="lat", time_col="datetime",
        days_window=1,  # ±1 day to catch same-day S2 passes
        neighborhood_px=1,
        scale_m=10,
        bands_to_sample=['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'],
        verbose=False
    )

    # Step 3: Load model and predict
    print("\n[3/3] Loading model and making predictions...")
    bundle = joblib.load(model_path)
    model = bundle['model']
    raw_inputs = bundle['raw_inputs']
    feature_names = bundle['feature_names']
    transform = bundle.get('transform', 'none')
    best_iteration = bundle.get('best_iteration', None)

    # Build feature matrix
    X = pd.get_dummies(df_s2.reindex(columns=raw_inputs), drop_first=False)
    X = X.reindex(columns=feature_names, fill_value=0)

    # Predict
    if best_iteration is not None:
        try:
            yhat = model.predict(X, iteration_range=(0, best_iteration + 1))
        except TypeError:
            yhat = model.predict(X)
    else:
        yhat = model.predict(X)

    # Inverse transform if needed
    if transform == 'log1p':
        yhat = np.expm1(yhat)

    df_s2['pred_Chla'] = yhat

    print(f"\n[predict_chla] Completed. Predictions made for {len(yhat)} rows.")
    print(f"  - Rows with S2 data: {df_s2['s2_found'].sum()}")
    print(f"  - Rows without S2: {(~df_s2['s2_found']).sum()}")
    print(f"  - Predictions range: [{yhat.min():.3f}, {yhat.max():.3f}]")

    return df_s2


# ============================================================================
# MAIN / TESTING
# ============================================================================

if __name__ == "__main__":
    # Example usage
    test_sites = [
        {"lat": 10.5, "lon": 75.5, "datetime": "2023-06-15 10:30:00"},
        {"lat": 10.6, "lon": 75.6, "datetime": "2023-06-16 11:00:00"},
    ]

    result = predict_chla(test_sites)
    print("\n=== RESULTS ===")
    print(result[['lat', 'lon', 'datetime', 's2_found', 'pred_Chla']].head(10))
