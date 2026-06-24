"""
Extract Google Earth Engine satellite features for each wildfire node.

For every fire (lat, lon, ignition date) we sample satellite rasters and attach
the reduced values as new node features for the Human-vs-Natural attribution task.

LEAKAGE GUARD: all time-varying signals (vegetation, weather, nightlights) are
sampled over a window STRICTLY BEFORE the ignition date. Sampling on/after the
fire would measure the burn scar / active-fire signal itself, which predicts
"a fire happened" rather than its cause. Static layers (terrain, population,
land cover) are time-invariant proxies and sampled as-is.

Feature families:
  - Human-presence : VIIRS nighttime lights, GHSL population & built surface, ESA land cover
  - Weather/drought: ERA5-Land precip sum, mean temp, dewpoint, 10m wind speed (-> VPD derived in pandas)
  - Vegetation/fuel: MODIS NDVI mean
  - Terrain        : SRTM elevation & slope

Usage:
    # one-time auth (opens a browser; needs a Cloud project with Earth Engine API enabled)
    earthengine authenticate

    .venv/bin/python satellite_features.py --project YOUR_GCP_PROJECT_ID

Output:
    data/WUMI2024a_wildfires_2020_2024_with_satellite.csv
        original fire columns + sat_* feature columns, ready to merge into
        data_preparation.py alongside log_area.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

try:
    import ee
except ImportError:  # pragma: no cover
    raise SystemExit(
        "earthengine-api is not installed. Run:\n"
        "    .venv/bin/pip install earthengine-api\n"
        "then `earthengine authenticate` once before running this script."
    )

ERA5_SCALE = 11132  # ERA5-Land nominal resolution (m)


def init_ee(project: str | None) -> None:
    """Initialize Earth Engine, prompting for auth on first run."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def load_fires(path: str, start_year: int, end_year: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["cause"] = df["cause_human_or_natural"].astype(str).str.strip().str.title()
    df = df[df["cause"].isin(["Human", "Natural"])]
    df = df.dropna(subset=["lat", "lon", "year", "month", "day"])
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    df = df.reset_index(drop=True)
    return df


def build_extractor(window_days: int):
    """Return a server-side function mapping a fire point -> point + sat features."""

    def extract(feat: "ee.Feature") -> "ee.Feature":
        pt = feat.geometry()
        d = ee.Date(feat.get("sat_date"))
        start_win = d.advance(-window_days, "day")
        start_year = d.advance(-365, "day")

        # --- Vegetation / fuel: MODIS NDVI (250 m), scaled to [-1, 1] ---
        ndvi = (
            ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate(start_win, d)
            .select("NDVI")
            .mean()
            .multiply(0.0001)
            .rename("sat_ndvi")
        )

        # --- Weather / drought: ERA5-Land daily aggregates ---
        era = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start_win, d)
        precip = era.select("total_precipitation_sum").sum().rename("sat_precip_m")
        temp = era.select("temperature_2m").mean().rename("sat_temp_k")
        dew = era.select("dewpoint_temperature_2m").mean().rename("sat_dewpoint_k")
        u = era.select("u_component_of_wind_10m").mean()
        v = era.select("v_component_of_wind_10m").mean()
        wind = u.hypot(v).rename("sat_wind_ms")

        # --- Human presence: VIIRS nighttime lights, 1-year pre-fire mean ---
        viirs = (
            ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
            .filterDate(start_year, d)
            .select("avg_rad")
            .mean()
            .rename("sat_nightlights")
        )

        # --- Static layers (time-invariant) ---
        srtm = ee.Image("USGS/SRTMGL1_003")
        elevation = srtm.select("elevation").rename("sat_elevation")
        slope = ee.Terrain.slope(srtm).rename("sat_slope")
        pop = ee.Image("JRC/GHSL/P2023A/GHS_POP/2020").select("population_count").rename("sat_pop")
        built = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020").select("built_surface").rename("sat_built")
        landcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("sat_landcover")

        def rr(img, scale, reducer=None):
            reducer = reducer or ee.Reducer.mean()
            return img.reduceRegion(reducer=reducer, geometry=pt, scale=scale, maxPixels=1e9)

        out = (
            rr(ndvi, 250)
            .combine(rr(precip, ERA5_SCALE))
            .combine(rr(temp, ERA5_SCALE))
            .combine(rr(dew, ERA5_SCALE))
            .combine(rr(wind, ERA5_SCALE))
            .combine(rr(viirs, 500))
            .combine(rr(elevation, 30))
            .combine(rr(slope, 30))
            .combine(rr(pop, 100))
            .combine(rr(built, 100))
            .combine(rr(landcover, 10, ee.Reducer.mode()))
        )
        return ee.Feature(None, out).set("fireid", feat.get("fireid"))

    return extract


def to_features(df: pd.DataFrame) -> list:
    millis = (df["date"].astype("int64") // 10**6).tolist()
    lat, lon, fid = df["lat"].tolist(), df["lon"].tolist(), df["fireid"].tolist()
    return [
        ee.Feature(ee.Geometry.Point([lon[i], lat[i]]), {"fireid": fid[i], "sat_date": int(millis[i])})
        for i in range(len(df))
    ]


def run_chunk(features: list, extract, retries: int = 4) -> list[dict]:
    fc = ee.FeatureCollection(features).map(extract)
    for attempt in range(retries):
        try:
            info = fc.getInfo()
            return [f["properties"] for f in info["features"]]
        except Exception as e:  # transient EE errors / rate limits
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"    chunk failed ({e}); retrying in {wait}s...")
            time.sleep(wait)
    return []


def derive_vpd(df: pd.DataFrame) -> pd.DataFrame:
    """Vapour pressure deficit (kPa) from mean temp & dewpoint (Tetens)."""

    def es(t_celsius):
        return 0.6108 * np.exp(17.27 * t_celsius / (t_celsius + 237.3))

    if "sat_temp_k" in df and "sat_dewpoint_k" in df:
        tc = df["sat_temp_k"] - 273.15
        td = df["sat_dewpoint_k"] - 273.15
        df["sat_vpd_kpa"] = (es(tc) - es(td)).clip(lower=0)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/WUMI2024a_wildfires_1984_2024_with_subfires.txt")
    ap.add_argument("--out", default="data/WUMI2024a_wildfires_2020_2024_with_satellite.csv")
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"),
                    help="Google Cloud project id with Earth Engine API enabled (or set EE_PROJECT)")
    ap.add_argument("--start-year", type=int, default=2020)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--window-days", type=int, default=90,
                    help="pre-ignition window for time-varying signals")
    ap.add_argument("--chunk-size", type=int, default=200)
    args = ap.parse_args()

    init_ee(args.project)

    fires = load_fires(args.data, args.start_year, args.end_year)
    print(f"Extracting satellite features for {len(fires)} fires ({args.start_year}-{args.end_year}), "
          f"{args.window_days}-day pre-ignition window")

    extract = build_extractor(args.window_days)
    features = to_features(fires)

    rows: list[dict] = []
    n_chunks = (len(features) + args.chunk_size - 1) // args.chunk_size
    for ci in range(n_chunks):
        chunk = features[ci * args.chunk_size:(ci + 1) * args.chunk_size]
        print(f"  chunk {ci + 1}/{n_chunks} ({len(chunk)} points)...")
        rows.extend(run_chunk(chunk, extract))

    sat = pd.DataFrame(rows)
    sat = derive_vpd(sat)

    merged = fires.merge(sat, on="fireid", how="left", validate="one_to_one")
    sat_cols = [c for c in merged.columns if c.startswith("sat_")]
    missing = merged[sat_cols].isna().any(axis=1).sum()
    print(f"Done. {len(merged)} rows, {len(sat_cols)} sat features; "
          f"{missing} rows with >=1 missing sat value.")
    merged.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
