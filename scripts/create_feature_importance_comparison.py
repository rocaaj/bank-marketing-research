"""
create_feature_importance_comparison.py
-----------------------------------
Create feature importance comparison visualizations across all models.

Generates:
- Plot B: Top 10 Feature Comparison Chart (horizontal bar chart with multiple bars per feature)
- Plot C: Feature Importance Heatmap (top 15 features across all models)

Outputs:
- v2_top10_features_comparison.png (Plot B)
- v2_feature_importance_heatmap.png (Plot C)
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "output" / "classification"

# File paths
FILE_PATHS = {
    'Logistic Regression': OUTPUT_PATH / "v2_lr_coefficients.csv",
    'LightGBM': OUTPUT_PATH / "v2_lgb_shap_importance.csv",
    'XGBoost': OUTPUT_PATH / "v2_xgb_shap_importance.csv",  # May not exist
    'CatBoost': OUTPUT_PATH / "v2_cat_shap_importance.csv",
}

# Model colors
MODEL_COLORS = {
    'Logistic Regression': '#e74c3c',  # Red
    'LightGBM': '#3498db',              # Blue
    'XGBoost': '#2ecc71',               # Green
    'CatBoost': '#f39c12',              # Orange
}


def load_feature_importance():
    """Load feature importance data from all available models."""
    print("🔄 Loading feature importance data from all models...")
    
    importance_data = {}
    
    for model_name, file_path in FILE_PATHS.items():
        if not file_path.exists():
            print(f"   ⚠️  Skipping {model_name}: {file_path.name} not found")
            continue
        
        try:
            df = pd.read_csv(file_path)
            
            # Handle different file formats
            if 'abs_coefficient' in df.columns:
                # Logistic Regression: use abs_coefficient
                df_model = df[['feature', 'abs_coefficient']].copy()
                df_model.columns = ['feature', 'importance']
            elif 'importance' in df.columns:
                # SHAP importance files
                df_model = df[['feature', 'importance']].copy()
            elif 'coefficient' in df.columns:
                # Fallback: use absolute coefficient
                df_model = df[['feature', 'coefficient']].copy()
                df_model['importance'] = df_model['coefficient'].abs()
                df_model = df_model[['feature', 'importance']]
            else:
                print(f"   ⚠️  {model_name}: Unknown file format")
                continue
            
            # Normalize importance to 0-1 scale for comparison
            max_importance = df_model['importance'].max()
            if max_importance > 0:
                df_model['importance_normalized'] = df_model['importance'] / max_importance
            else:
                df_model['importance_normalized'] = 0
            
            importance_data[model_name] = df_model.sort_values('importance', ascending=False)
            print(f"   ✅ Loaded {model_name}: {len(df_model)} features")
            
        except Exception as e:
            print(f"   ⚠️  Error loading {model_name}: {e}")
            continue
    
    if len(importance_data) == 0:
        raise ValueError("No feature importance data was successfully loaded!")
    
    print(f"\n✅ Successfully loaded {len(importance_data)} model(s)")
    return importance_data


def get_top_features_unified(importance_data, top_n=15):
    """Get top features across all models, unified by feature name."""
    print(f"\n🔄 Identifying top {top_n} features across all models...")
    
    # Collect all features and their max importance across models
    feature_scores = {}
    
    for model_name, df in importance_data.items():
        top_features = df.head(top_n)
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            if feature not in feature_scores:
                feature_scores[feature] = {}
            feature_scores[feature][model_name] = importance
    
    # Get top N features by maximum importance across all models
    feature_max_scores = {
        feat: max(scores.values()) if scores else 0
        for feat, scores in feature_scores.items()
    }
    
    top_features = sorted(feature_max_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_feature_names = [feat for feat, _ in top_features]
    
    print(f"✅ Identified {len(top_feature_names)} top features")
    return top_feature_names


def create_top10_comparison_chart(importance_data):
    """Create Plot B: Top 10 Feature Comparison Chart."""
    print("\n🔄 Creating Top 10 Feature Comparison Chart (Plot B)...")
    
    # Get top 10 features
    top_features = get_top_features_unified(importance_data, top_n=10)
    
    # Prepare data for plotting
    plot_data = []
    for feature in top_features:
        for model_name, df in importance_data.items():
            feature_row = df[df['feature'] == feature]
            if len(feature_row) > 0:
                importance = feature_row.iloc[0]['importance']
                plot_data.append({
                    'feature': feature,
                    'model': model_name,
                    'importance': importance
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Normalize importance within each model for better comparison
    normalized_plot_data = []
    for model_name in plot_df['model'].unique():
        model_data = plot_df[plot_df['model'] == model_name].copy()
        max_importance = model_data['importance'].max()
        if max_importance > 0:
            model_data['importance_norm'] = model_data['importance'] / max_importance
        else:
            model_data['importance_norm'] = 0
        normalized_plot_data.append(model_data)
    
    plot_df_norm = pd.concat(normalized_plot_data, ignore_index=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up positions for grouped bars
    features = top_features[::-1]  # Reverse for top-to-bottom display
    n_features = len(features)
    n_models = len(importance_data)
    bar_width = 0.8 / n_models
    
    x_pos = np.arange(n_features)
    
    # Plot bars for each model
    for idx, (model_name, color) in enumerate(MODEL_COLORS.items()):
        if model_name not in importance_data:
            continue
        
        model_importances = []
        for feature in features:
            feature_data = plot_df_norm[
                (plot_df_norm['feature'] == feature) & 
                (plot_df_norm['model'] == model_name)
            ]
            if len(feature_data) > 0:
                model_importances.append(feature_data.iloc[0]['importance_norm'])
            else:
                model_importances.append(0)
        
        offset = (idx - (n_models - 1) / 2) * bar_width
        ax.barh(x_pos + offset, model_importances, bar_width, 
                label=model_name, color=color, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Formatting
    ax.set_yticks(x_pos)
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel('Normalized Feature Importance', fontsize=13, fontweight='bold')
    ax.set_title('Top 10 Features: Importance Comparison Across Models', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1.1])
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_PATH / "v2_top10_features_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved Plot B: {output_path}")
    return output_path


def create_feature_importance_heatmap(importance_data):
    """Create Plot C: Feature Importance Heatmap."""
    print("\n🔄 Creating Feature Importance Heatmap (Plot C)...")
    
    # Get top 15 features
    top_features = get_top_features_unified(importance_data, top_n=15)
    
    # Create matrix: rows = features, columns = models
    heatmap_data = []
    
    for feature in top_features:
        row = {'feature': feature}
        for model_name in importance_data.keys():
            df = importance_data[model_name]
            feature_row = df[df['feature'] == feature]
            if len(feature_row) > 0:
                # Use normalized importance for heatmap
                row[model_name] = feature_row.iloc[0]['importance_normalized']
            else:
                row[model_name] = 0
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df = heatmap_df.set_index('feature')
    
    # Reverse order so top feature is at top
    heatmap_df = heatmap_df.reindex(top_features[::-1])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create heatmap with custom colormap
    sns.heatmap(heatmap_df, 
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Normalized Importance (0-1)'},
                linewidths=0.5,
                linecolor='white',
                ax=ax,
                vmin=0,
                vmax=1)
    
    ax.set_title('Feature Importance Heatmap: Top 15 Features Across Models', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=13, fontweight='bold')
    
    # Rotate feature names for readability
    plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_PATH / "v2_feature_importance_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved Plot C: {output_path}")
    return output_path


def print_feature_consensus(importance_data):
    """Print a summary of feature consensus across models."""
    print("\n" + "=" * 80)
    print("FEATURE CONSENSUS ANALYSIS")
    print("=" * 80)
    
    # Get top 10 features per model
    top_features_by_model = {}
    for model_name, df in importance_data.items():
        top_features_by_model[model_name] = df.head(10)['feature'].tolist()
    
    # Find features that appear in all models' top 10
    all_models = list(top_features_by_model.keys())
    if len(all_models) > 1:
        consensus_features = set(top_features_by_model[all_models[0]])
        for model_name in all_models[1:]:
            consensus_features = consensus_features.intersection(
                set(top_features_by_model[model_name])
            )
        
        print(f"\n📊 Features in ALL models' top 10 (Consensus Features):")
        for idx, feature in enumerate(sorted(consensus_features, 
                                             key=lambda x: sum([
                                                 df[df['feature'] == x]['importance'].iloc[0] 
                                                 if len(df[df['feature'] == x]) > 0 else 0
                                                 for df in importance_data.values()
                                             ]), reverse=True), 1):
            print(f"   {idx}. {feature}")
    
    print("\n📊 Top 5 Features by Model:")
    for model_name, df in importance_data.items():
        print(f"\n   {model_name}:")
        for idx, row in df.head(5).iterrows():
            print(f"      {idx + 1}. {row['feature']} (importance: {row['importance']:.4f})")
    
    print("\n" + "=" * 80)


def main():
    """Main execution function."""
    print("=" * 80)
    print("FEATURE IMPORTANCE COMPARISON GENERATOR")
    print("=" * 80)
    
    # Ensure output directory exists
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load feature importance data
        importance_data = load_feature_importance()
        
        # Print consensus analysis
        print_feature_consensus(importance_data)
        
        # Create Plot B: Top 10 Comparison Chart
        plot_b_path = create_top10_comparison_chart(importance_data)
        
        # Create Plot C: Feature Importance Heatmap
        plot_c_path = create_feature_importance_heatmap(importance_data)
        
        print("\n" + "=" * 80)
        print("VISUALIZATION GENERATION COMPLETE")
        print("=" * 80)
        print(f"\n✅ Plot B (Top 10 Comparison): {plot_b_path.name}")
        print(f"✅ Plot C (Heatmap): {plot_c_path.name}")
        print(f"\nAll outputs saved in: {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

