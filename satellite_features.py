"""
Extract Google Earth Engine satellite features across three distinct time windows:
  1. 3-Day Window  : Acute Fire Weather (Ignition & Spread conditions)
  2. 30-Day Window : Fuel Curing Trend (Short-term drying/heatwaves)
  3. 90-Day Window : Seasonal Accumulation (Long-term background climate/biomass growth)

Output:
    Three separate CSV files in ./data/ corresponding to each window timeline.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

try:
    import ee
except ImportError:
    raise SystemExit("earthengine-api is not installed. Run: pip install earthengine-api")

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


def build_extractor(start_year_int: int, end_year_int: int):
    """Return an optimized server-side function mapping a fire point -> properties.
    Collections are loaded once globally and pre-filtered to optimize evaluation trees.
    """
    global_start = ee.Date(f"{start_year_int - 1}-01-01")
    global_end = ee.Date(f"{end_year_int + 1}-01-01")

    # Pre-filtered background collections to speed up search trees
    base_ndvi = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(global_start, global_end)
    base_aod = ee.ImageCollection("MODIS/061/MCD19A2_GRANULES").filterDate(global_start, global_end)
    base_era = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(global_start, global_end)
    base_viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG").filterDate(global_start, global_end)

    # Static Assets
    srtm = ee.Image("USGS/SRTMGL1_003")
    elevation = srtm.select("elevation").rename("sat_elevation")
    slope = ee.Terrain.slope(srtm).rename("sat_slope")
    pop = ee.Image("JRC/GHSL/P2023A/GHS_POP/2020").select("population_count").rename("sat_pop")
    built = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020").select("built_surface").rename("sat_built")
    landcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("sat_landcover")

    # Clean Fallback Dummy Rasters
    dummy_ndvi = ee.Image.constant(0).rename("NDVI").selfMask()
    dummy_aod = ee.Image.constant(0).rename("Optical_Depth_047").selfMask()
    dummy_nl = ee.Image.constant(0).rename("avg_rad").selfMask()

    def extract(feat: "ee.Feature") -> "ee.Feature":
        pt = feat.geometry()
        d = ee.Date(feat.get("sat_date"))
        
        start_year = d.advance(-365, "day")
        w3 = d.advance(-3, "day")
        w30 = d.advance(-30, "day")
        w90 = d.advance(-90, "day")

        def safe_mean(collection, band_name, dummy_img):
            selected = collection.select(band_name)
            return ee.Image(ee.Algorithms.If(selected.size().gt(0), selected.mean(), dummy_img))

        # ==========================================
        # 1. VEGETATION & AOD FILTERS
        # ==========================================
        ndvi_3d = safe_mean(base_ndvi.filterDate(w3, d), "NDVI", dummy_ndvi).multiply(0.0001).rename("sat_ndvi_3d")
        ndvi_30d = safe_mean(base_ndvi.filterDate(w30, d), "NDVI", dummy_ndvi).multiply(0.0001).rename("sat_ndvi_30d")
        ndvi_90d = safe_mean(base_ndvi.filterDate(w90, d), "NDVI", dummy_ndvi).multiply(0.0001).rename("sat_ndvi_90d")

        aod_3d = safe_mean(base_aod.filterDate(w3, d), "Optical_Depth_047", dummy_aod).multiply(0.001).rename("sat_aod_3d")
        aod_30d = safe_mean(base_aod.filterDate(w30, d), "Optical_Depth_047", dummy_aod).multiply(0.001).rename("sat_aod_30d")
        aod_90d = safe_mean(base_aod.filterDate(w90, d), "Optical_Depth_047", dummy_aod).multiply(0.001).rename("sat_aod_90d")

        # ==========================================
        # 2. METEOROLOGY WINDOW FILTERS
        # ==========================================
        era_3d = base_era.filterDate(w3, d)
        precip_3d = era_3d.select("total_precipitation_sum").sum().rename("sat_precip_m_3d")
        temp_3d = era_3d.select("temperature_2m").mean().rename("sat_temp_k_3d")
        dew_3d = era_3d.select("dewpoint_temperature_2m").mean().rename("sat_dewpoint_k_3d")
        wind_3d = era_3d.select("u_component_of_wind_10m").mean().hypot(era_3d.select("v_component_of_wind_10m").mean()).rename("sat_wind_ms_3d")

        era_30d = base_era.filterDate(w30, d)
        precip_30d = era_30d.select("total_precipitation_sum").sum().rename("sat_precip_m_30d")
        temp_30d = era_30d.select("temperature_2m").mean().rename("sat_temp_k_30d")
        dew_30d = era_30d.select("dewpoint_temperature_2m").mean().rename("sat_dewpoint_k_30d")
        wind_30d = era_30d.select("u_component_of_wind_10m").mean().hypot(era_30d.select("v_component_of_wind_10m").mean()).rename("sat_wind_ms_30d")

        era_90d = base_era.filterDate(w90, d)
        precip_90d = era_90d.select("total_precipitation_sum").sum().rename("sat_precip_m_90d")
        temp_90d = era_90d.select("temperature_2m").mean().rename("sat_temp_k_90d")
        dew_90d = era_90d.select("dewpoint_temperature_2m").mean().rename("sat_dewpoint_k_90d")
        wind_90d = era_90d.select("u_component_of_wind_10m").mean().hypot(era_90d.select("v_component_of_wind_10m").mean()).rename("sat_wind_ms_90d")

        # ==========================================
        # 3. CONTEXTUAL & STATIC ENVIRONMENT
        # ==========================================
        viirs = safe_mean(base_viirs.filterDate(start_year, d), "avg_rad", dummy_nl).rename("sat_nightlights")

        # OPTIMIZED SCALE ASSIGNMENT: Prevents GEE from hanging on hyper-local sub-pixel operations
        def rr(img, scale, reducer=None):
            reducer = reducer or ee.Reducer.mean()
            optimized_scale = 100 if scale < 100 else scale
            return img.reduceRegion(reducer=reducer, geometry=pt, scale=optimized_scale, maxPixels=1e9)

        out = (
            rr(ndvi_3d, 250).combine(rr(ndvi_30d, 250)).combine(rr(ndvi_90d, 250))
            .combine(rr(aod_3d, 1000)).combine(rr(aod_30d, 1000)).combine(rr(aod_90d, 1000))
            .combine(rr(precip_3d, ERA5_SCALE)).combine(rr(precip_30d, ERA5_SCALE)).combine(rr(precip_90d, ERA5_SCALE))
            .combine(rr(temp_3d, ERA5_SCALE)).combine(rr(temp_30d, ERA5_SCALE)).combine(rr(temp_90d, ERA5_SCALE))
            .combine(rr(dew_3d, ERA5_SCALE)).combine(rr(dew_30d, ERA5_SCALE)).combine(rr(dew_90d, ERA5_SCALE))
            .combine(rr(wind_3d, ERA5_SCALE)).combine(rr(wind_30d, ERA5_SCALE)).combine(rr(wind_90d, ERA5_SCALE))
            .combine(rr(viirs, 500))
            .combine(rr(elevation, 100))
            .combine(rr(slope, 100))
            .combine(rr(pop, 100))
            .combine(rr(built, 100))
            .combine(rr(landcover, 100, ee.Reducer.mode()))
        )
        return ee.Feature(None, out).set("fireid", feat.get("fireid"))

    return extract


def to_features(df: pd.DataFrame) -> list:
    date_strs = df["date"].dt.strftime("%Y-%m-%d").tolist()
    lat, lon, fid = df["lat"].tolist(), df["lon"].tolist(), df["fireid"].tolist()
    return [
        ee.Feature(ee.Geometry.Point([lon[i], lat[i]]), {"fireid": fid[i], "sat_date": date_strs[i]})
        for i in range(len(df))
    ]


def run_chunk(features: list, extract, retries: int = 5) -> list[dict]:
    fc = ee.FeatureCollection(features).map(extract)
    for attempt in range(retries):
        try:
            info = fc.getInfo()
            return [f["properties"] for f in info["features"]]
        except Exception as e:
            if attempt == retries - 1:
                print(f"\n[ERROR] Permanent failure on chunk batch: {e}")
                return []
            time.sleep(2 ** attempt * 4)
    return []


def derive_vpd_for_window(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    def es(t_celsius):
        return 0.6108 * np.exp(17.27 * t_celsius / (t_celsius + 237.3))

    temp_col = f"sat_temp_k_{suffix}"
    dew_col = f"sat_dewpoint_k_{suffix}"
    vpd_col = f"sat_vpd_kpa_{suffix}"

    if temp_col in df and dew_col in df:
        tc = df[temp_col] - 273.15
        td = df[dew_col] - 273.15
        df[vpd_col] = (es(tc) - es(td)).clip(lower=0)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/WUMI2024a_wildfires_1984_2024_with_subfires.txt")
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    ap.add_argument("--start-year", type=int, default=2020)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--chunk-size", type=int, default=50)
    args = ap.parse_args()

    init_ee(args.project)
    os.makedirs("data", exist_ok=True)

    fires = load_fires(args.data, args.start_year, args.end_year)
    print(f"Extracting features for {len(fires)} fires ({args.start_year}-{args.end_year})...")

    extract = build_extractor(args.start_year, args.end_year)
    features = to_features(fires)

    rows: list[dict] = []
    n_chunks = (len(features) + args.chunk_size - 1) // args.chunk_size
    
    # Clean sequential loops with reliable tracking
    for ci in range(n_chunks):
        chunk = features[ci * args.chunk_size:(ci + 1) * args.chunk_size]
        print(f"Processing Batch {ci + 1}/{n_chunks} ({len(chunk)} points)... ", end="", flush=True)
        
        start_time = time.time()
        chunk_data = run_chunk(chunk, extract)
        rows.extend(chunk_data)
        
        elapsed = time.time() - start_time
        print(f"Done! (Took {elapsed:.1f}s) | Total Extracted: {len(rows)}/{len(features)}")

    sat_master = pd.DataFrame(rows)
    if sat_master.empty:
        print("Fatal error: Extracted dataset matrix completely empty.")
        return

    static_cols = ['sat_nightlights', 'sat_elevation', 'sat_slope', 'sat_pop', 'sat_built', 'sat_landcover']
    windows_config = {
        "3d": ['sat_ndvi_3d', 'sat_aod_3d', 'sat_precip_m_3d', 'sat_temp_k_3d', 'sat_dewpoint_k_3d', 'sat_wind_ms_3d', 'sat_vpd_kpa_3d'],
        "30d": ['sat_ndvi_30d', 'sat_aod_30d', 'sat_precip_m_30d', 'sat_temp_k_30d', 'sat_dewpoint_k_30d', 'sat_wind_ms_30d', 'sat_vpd_kpa_30d'],
        "90d": ['sat_ndvi_90d', 'sat_aod_90d', 'sat_precip_m_90d', 'sat_temp_k_90d', 'sat_dewpoint_k_90d', 'sat_wind_ms_90d', 'sat_vpd_kpa_90d']
    }
    
    all_expected_cols = ['fireid'] + static_cols + windows_config["3d"] + windows_config["30d"] + windows_config["90d"]
    for col in all_expected_cols:
        if col not in sat_master.columns:
            sat_master[col] = np.nan

    print("\nCalculating metrics and sorting output panels...")
    for window in ["3d", "30d", "90d"]:
        sat_master = derive_vpd_for_window(sat_master, window)

    for name, dynamic_cols in windows_config.items():
        target_features = sat_master[['fireid'] + dynamic_cols + static_cols]
        merged_window = fires.merge(target_features, on="fireid", how="left", validate="one_to_one")
        rename_map = {col: col.replace(f"_{name}", "") for col in dynamic_cols}
        merged_window = merged_window.rename(columns=rename_map)
        
        out_path = f"data/WUMI2024a_wildfires_2020_2024_{name}_satellite.csv"
        merged_window.to_csv(out_path, index=False)
        print(f"Saved Output -> {out_path}")


if __name__ == "__main__":
    main()