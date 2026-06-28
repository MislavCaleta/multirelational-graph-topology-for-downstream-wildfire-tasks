import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Setup paths and directories
output_dir = "./outputs/figures"
os.makedirs(output_dir, exist_ok=True)

# Define the multi-window datasets to process
datasets_config = {
    "3d": {"path": "./data/WUMI2024a_wildfires_2020_2024_3d_satellite.csv", "label": "3-Day Window (Acute Weather)"},
    "30d": {"path": "./data/WUMI2024a_wildfires_2020_2024_30d_satellite.csv", "label": "30-Day Window (Fuel Curing)"},
    "90d": {"path": "./data/WUMI2024a_wildfires_2020_2024_90d_satellite.csv", "label": "90-Day Window (Seasonal Growth)"}
}

# Apply User-defined Figure Configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'axes.linewidth': 1.5,
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})
sns.set_theme(style="whitegrid", font="serif")

# LOCKED FIXED ORDER: This order will never change across any graph
fixed_features_order = [
    'sat_vpd_kpa', 'sat_temp_k', 'sat_wind_ms', 'sat_dewpoint_k', 'sat_precip_m', 'sat_ndvi',
    'sat_aod', 'sat_slope', 'sat_elevation', 'sat_nightlights', 'sat_built', 'sat_pop'
]
target = 'log_area_ha'

for window_key, config in datasets_config.items():
    if not os.path.exists(config["path"]):
        print(f"Skipping {window_key}: File not found at {config['path']}")
        continue
        
    print(f"Processing structural locked figures for {config['label']}...")
    df = pd.read_csv(config["path"])

    # Establish target metrics
    if 'burn_area_ha' in df.columns and df['burn_area_ha'].notna().sum() > 0:
        df[target] = np.log10(df['burn_area_ha'] + 1)
    elif 'poly_area_ha' in df.columns:
        df[target] = np.log10(df['poly_area_ha'] + 1)
    else:
        print(f"Error: No valid area target found in {config['path']}. Skipping.")
        continue

    # Identify features completely missing data in the raw file before fillna
    missing_data_features = [f for f in fixed_features_order if f not in df.columns or df[f].isna().all()]

    # Impute missing entries with the column median per slice to prevent calculation breaks
    for f in fixed_features_order:
        if f in df.columns:
            if df[f].isna().all():
                df[f] = 0.0
            else:
                df[f] = df[f].fillna(df[f].median())
        else:
            df[f] = 0.0  # Force initialization if entirely absent from file index

    # Storage dictionary to cleanly print text summaries after generating plots
    text_summary_data = {"Human": {}, "Natural": {}}

    # ----------------------------------------------------
    # GENERATE FIXED SPLIT FIGURES (HUMAN VS NATURAL)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharex=True)
    
    causes = ['Human', 'Natural']
    base_colors = {'Human': '#777777', 'Natural': '#4285F4'} # Professional Grey vs Blue
    alert_red = '#EA4335' # Bright Red for missing data flags
    
    for idx, cause_type in enumerate(causes):
        ax = axes[idx]
        df_sub = df[df['cause'] == cause_type]
        
        if len(df_sub) < 5:
            ax.text(0.5, 0.5, f"Insufficient data for {cause_type}", ha='center', va='center')
            continue
            
        # Compute subset specific correlation matrix
        sub_corr = df_sub[fixed_features_order + [target]].corr().fillna(0.0)
        
        # Build DataFrame mapped to the strict locked order
        area_corr = pd.DataFrame({
            'Variable': fixed_features_order,
            'Correlation': [sub_corr.loc[f, target] for f in fixed_features_order]
        })
        
        # Populate text storage dictionary for the terminal output block below
        for f in fixed_features_order:
            val = sub_corr.loc[f, target]
            flag = " [MISSING DATA]" if f in missing_data_features else ""
            text_summary_data[cause_type][f] = f"{val:.4f}{flag}"

        # Map out bar colors: normal thematic color vs alert red for omitted parameters
        bar_colors = [
            alert_red if var in missing_data_features else base_colors[cause_type]
            for var in area_corr['Variable']
        ]
        
        sns.barplot(
            data=area_corr, 
            x='Correlation', 
            y='Variable', 
            palette=bar_colors,
            edgecolor='black',
            linewidth=1.2,
            ax=ax
        )
        
        # Visual cross-out/label overlay directly on variables lacking sufficient data rows
        for label in ax.get_yticklabels():
            if label.get_text() in missing_data_features:
                label.set_color('#D93025')
                label.set_weight('bold')
                
        ax.set_title(f"{cause_type} Causation Matrix ($n$={len(df_sub)})", fontsize=22, weight='bold', pad=15)
        ax.set_xlabel("Pearson Correlation Coefficient ($r$)", labelpad=12)
        ax.set_ylabel("", labelpad=0) 
        ax.set_xlim(-0.5, 0.5)  # Constrain axis scale uniformly to view balance changes directly
        
        if idx > 0:
            ax.set_yticklabels([]) # Clean duplicate labels off right panel

    plt.suptitle(f"Aligned Variable Splits with Log Burned Area\n({config['label']})", fontsize=24, weight='bold', y=1.02)
    plt.tight_layout()
    
    fig_split_path = os.path.join(output_dir, f"{window_key}_human_vs_natural_correlations.png")
    plt.savefig(fig_split_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved Aligned Panel -> {fig_split_path}")

    # ====================================================
    # TEXT TERMINAL OUTPUT COPIER BLOCK
    # ====================================================
    print("\n" + "="*60)
    print(f"CORRELATION DATA EXPORT: {config['label'].upper()}")
    print("="*60)
    print(f"{'Feature Variable':<20} | {'Human Cause (r)':<20} | {'Natural Cause (r)':<20}")
    print("-"*60)
    for feat in fixed_features_order:
        h_val = text_summary_data["Human"].get(feat, "N/A")
        n_val = text_summary_data["Natural"].get(feat, "N/A")
        print(f"{feat:<20} | {h_val:<20} | {n_val:<20}")
    print("="*60 + "\n")

print("All comparative visualizations and terminal summaries rendered successfully.")