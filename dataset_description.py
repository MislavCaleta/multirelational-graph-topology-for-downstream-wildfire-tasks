import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

# Set publication-quality plotting styles (IEEE standard look)
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'savefig.dpi': 300,
    'figure.autolayout': True
})
sns.set_palette("colorblind")

def load_and_preprocess(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Initial row count: {len(df)}")
    
    # 1. Preprocessing protocol: drop rows missing core attributes
    core_columns = ['lat', 'lon', 'year', 'poly_area_ha', 'cause_human_or_natural']
    df_clean = df.dropna(subset=core_columns).copy()
    
    # Remove invalid or undefined labels
    df_clean = df_clean[df_clean['cause_human_or_natural'].isin(['Human', 'Natural'])]
    
    # 2. Compute log-transformed burned area: ln(Area_ha + 1)
    df_clean['log_poly_area'] = np.log1p(df_clean['poly_area_ha'])
    
    # 3. Create binary target column (Human = 1, Natural = 0)
    df_clean['label'] = df_clean['cause_human_or_natural'].apply(lambda x: 1 if x == 'Human' else 0)
    
    print(f"Cleaned row count: {len(df_clean)}")
    return df_clean

def generate_statistics(df):
    print("\n" + "="*40)
    print("        DATASET STATISTICAL SUMMARY        ")
    print("="*40)
    
    total_fires = len(df)
    human_count = (df['label'] == 1).sum()
    natural_count = (df['label'] == 0).sum()
    
    print(f"Total valid fire records: {total_fires}")
    print(f"Anthropogenic (Human) Ignitions (Label 1): {human_count} ({human_count/total_fires*100:.2f}%)")
    print(f"Natural Ignitions (Label 0): {natural_count} ({natural_count/total_fires*100:.2f}%)")
    
    print("\n--- Temporal Range ---")
    print(f"Years covered: {int(df['year'].min())} to {int(df['year'].max())}")
    
    print("\n--- Geospatial Extent ---")
    print(f"Latitude bounds:  [{df['lat'].min():.4f}, {df['lat'].max():.4f}]")
    print(f"Longitude bounds: [{df['lon'].min():.4f}, {df['lon'].max():.4f}]")
    
    print("\n--- Burned Area Statistics (ha) ---")
    print(df['poly_area_ha'].describe())
    
    print("\n--- Log-Transformed Area Statistics [ln(Area + 1)] ---")
    print(df['log_poly_area'].describe())
    
    if 'mtbs_ID' in df.columns:
        unique_large_fires = df['mtbs_ID'].dropna().nunique()
        print(f"\nUnique MTBS Parent Complexes identified: {unique_large_fires}")
        
    print("="*40 + "\n")

def create_plots(df, output_dir="./outputs/figures"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # --- Plot 1: Geospatial Distribution Over Topographical Terrain Map ---
    print("Generating geospatial distribution map with shaded topographic relief basemap...")
    import contextily as ctx
    from shapely.geometry import box
    import matplotlib.patheffects as pe
    
    # Initialize the figure matching IEEE single/double-column constraints
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    
    # 1. Convert coordinates to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    
    # 2. Re-project to Web Mercator (EPSG:3857) to ensure tile mapping and elevation overlays don't warp
    gdf_merc = gdf.to_crs(epsg=3857)
    
    # Downsample points for rendering clarity so they don't completely bury the mountain/valley topography
    sample_size = min(len(gdf_merc), 25000)
    gdf_sample = gdf_merc.sample(sample_size, random_state=42)
    
    # 3. Plot the data points with a slightly lower alpha and smaller marker size
    sns.scatterplot(
        ax=ax,
        x=gdf_sample.geometry.x, 
        y=gdf_sample.geometry.y, 
        hue=gdf_sample['cause_human_or_natural'],
        alpha=0.55, 
        s=5.0, 
        palette={'Human': "#ff0404", 'Natural': "#00ff37"},
        zorder=2
    )
    
    # 4. Pull down the high-quality shaded relief terrain basemap underneath the scatter plot
    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldTerrain, zorder=1)
    except Exception as e:
        print(f"Primary terrain map timed out ({e}). Falling back to alternative terrain canvas...")
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.VoyagerNoLabels, zorder=1)
        except Exception as fallback_err:
            print(f"Could not load any basemap tile grids ({fallback_err}). Plotting points standalone.")

    # 5. Crop axes precisely to the Western US spatial box bounds (converted to Web Mercator limits)
    xmin, xmax = gdf_merc.geometry.x.min() - 50000, gdf_merc.geometry.x.max() + 50000
    ymin, ymax = gdf_merc.geometry.y.min() - 50000, gdf_merc.geometry.y.max() + 50000
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # 6. Stream and overlay comprehensive populated places using the high-resolution 1:50m scale file
    cities_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_populated_places_simple.geojson"
    try:
        print("Streaming high-resolution 1:50m city vectors...")
        cities = gpd.read_file(cities_url)
        cities_merc = cities.to_crs(epsg=3857)
        
        # Clip cities to your exact bounding box viewport
        bbox = box(xmin, ymin, xmax, ymax)
        visible_cities = cities_merc[cities_merc.geometry.within(bbox)].copy()
        
        # Plot EVERY single city/town center within the view as a solid black landmark dot
        ax.scatter(
            visible_cities.geometry.x,
            visible_cities.geometry.y,
            color='black',
            marker='o',
            s=14,
            edgecolor='white',
            linewidth=0.6,
            label='Populated Center',
            zorder=3
        )
        
        # To avoid text overlaps from hundreds of cities, filter labels systematically by importance
        # 'scalerank' up to 8 grabs major cities, state capitals, and important regional county hubs
        labeled_cities = visible_cities[visible_cities['scalerank'] <= 8]
        
        for idx, row in labeled_cities.iterrows():
            ax.text(
                row.geometry.x + 9000,  # Subtle text offset right
                row.geometry.y + 9000,  # Subtle text offset up
                row['name'],
                fontsize=7.5,
                fontweight='bold',
                color='#111111',
                zorder=4,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")]
            )
            
    except Exception as city_err:
        print(f"Warning: High-res city layers failed to compile ({city_err}).")
    
    ax.set_xlabel("Easting (Web Mercator meters)")
    ax.set_ylabel("Northing (Web Mercator meters)")
    ax.grid(True, linestyle='--', alpha=0.2, zorder=0)
    
    # Clean duplicates out of legend markers
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Ignition Cause", loc='lower left', frameon=True, facecolor='white', framealpha=0.95)
    
    plt.savefig(f"{output_dir}/fire_geographic_distribution.png", bbox_inches='tight', dpi=300)
    plt.close()

    # --- Plot 2: Yearly Fire Trends ---
    print("Generating annual frequency trends...")
    plt.figure(figsize=(7, 4))
    yearly_counts = df.groupby(['year', 'cause_human_or_natural']).size().unstack(fill_value=0)
    plt.plot(yearly_counts.index, yearly_counts['Natural'], label='Natural', color='#1b9e77', linewidth=2, marker='o', markersize=3)
    plt.plot(yearly_counts.index, yearly_counts['Human'], label='Anthropogenic', color='#d95f02', linewidth=2, marker='s', markersize=3)
    plt.xlabel("Year")
    plt.ylabel("Number of Ignitions")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(f"{output_dir}/annual_fire_trends.png", bbox_inches='tight')
    plt.close()

    # --- Plot 3: Distribution of Burned Area (Raw vs Log-Transformed) ---
    print("Generating size footprint histograms...")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    
    # Raw Distribution
    sns.histplot(df['poly_area_ha'], bins=50, ax=axes[0], color='purple', kde=False)
    axes[0].set_title("Raw Burned Area (ha)")
    axes[0].set_xlabel("Area (Hectares)")
    axes[0].set_ylabel("Count")
    axes[0].set_yscale('log') 
    
    # Log-Transformed Distribution
    sns.histplot(df['log_poly_area'], bins=30, ax=axes[1], color='teal', kde=True)
    axes[1].set_title(r"Log-Transformed $\ln(Area_{ha} + 1)$")
    axes[1].set_xlabel("Scaled Value")
    axes[1].set_ylabel("Count")
    
    plt.suptitle("Structural Metric Profiles of Wildfire Sizes", y=1.02)
    plt.savefig(f"{output_dir}/burned_area_distributions.png", bbox_inches='tight')
    plt.close()
    
    print(f"All verification plots successfully saved to the '{output_dir}/' directory.")

if __name__ == "__main__":
    dataset_path = "./data/WUMI2024a_wildfires_1984_2024_with_subfires.txt"
    
    if os.path.exists(dataset_path):
        processed_df = load_and_preprocess(dataset_path)
        generate_statistics(processed_df)
        create_plots(processed_df)
    else:
        print(f"Error: Dataset file '{dataset_path}' not found. Please double check file locations.")