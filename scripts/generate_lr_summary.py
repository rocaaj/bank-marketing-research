"""
generate_lr_summary.py
-----------------------------------
Generate comprehensive summary and interpretation for Logistic Regression results.
Outputs formatted tables and insights for presentation slide.

Outputs:
- v2_lr_coefficients.csv: Coefficient values and interpretations
- v2_lr_summary_table.csv: Summary metrics table
- v2_lr_slide_summary.txt: Formatted text summary for slide content
- v2_lr_top_features.png: Visualization of top features
"""

import sys
import pickle
import json
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
MODEL_PATH = OUTPUT_PATH / "models" / "v2_logistic_model.pkl"
THRESHOLD_ROI_PATH = OUTPUT_PATH / "v2_lr_threshold_roi.csv"
DATA_PATH = BASE_DIR / "data" / "bank-full-processed-v2.csv"
TARGET_COL = "y"

# Expected metrics from RESEARCH_LOG.md
EXPECTED_METRICS = {
    'roc_auc': 0.9133,
    'pr_auc': 0.5389,
    'precision_at_10': 0.5940,
    'lift_at_10': 5.08,
}


def load_model_and_data():
    """Load the saved logistic regression model and data."""
    print("🔄 Loading model and data...")
    
    # Load model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # Load data to get feature names
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    
    print(f"✅ Loaded model and data with {len(X.columns)} features")
    return model, X.columns


def extract_coefficients(model, feature_names):
    """Extract and interpret logistic regression coefficients."""
    print("\n🔄 Extracting coefficients...")
    
    # Handle CalibratedClassifierCV - extract estimator from calibrated_classifiers_
    if hasattr(model, 'calibrated_classifiers_') and len(model.calibrated_classifiers_) > 0:
        # Get the estimator from the first calibrated classifier
        base_estimator = model.calibrated_classifiers_[0].estimator
        print("   Detected CalibratedClassifierCV, extracting estimator from calibrated_classifiers_...")
    elif hasattr(model, 'estimator'):
        base_estimator = model.estimator
        print("   Using estimator attribute...")
    else:
        base_estimator = model
    
    # Handle Pipeline object
    if hasattr(base_estimator, 'named_steps') and 'model' in base_estimator.named_steps:
        actual_model = base_estimator.named_steps['model']
        print("   Extracting model from Pipeline...")
    elif hasattr(base_estimator, 'steps') and len(base_estimator.steps) > 0:
        # Alternative Pipeline structure
        actual_model = base_estimator.steps[-1][1]  # Get last step (the model)
        print("   Extracting model from Pipeline steps...")
    else:
        actual_model = base_estimator
    
    print(f"   Final model type: {type(actual_model)}")
    
    # Get coefficients
    if not hasattr(actual_model, 'coef_'):
        raise AttributeError(f"Model {type(actual_model)} does not have coef_ attribute. Available attributes: {[a for a in dir(actual_model) if not a.startswith('_')]}")
    
    coef = actual_model.coef_[0]
    intercept = actual_model.intercept_[0]
    
    # Create coefficient DataFrame
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef,
        'abs_coefficient': np.abs(coef),
    }).sort_values('abs_coefficient', ascending=False)
    
    # Calculate odds ratios (exp(coef))
    coef_df['odds_ratio'] = np.exp(coef_df['coefficient'])
    
    # Interpretation
    def interpret_coefficient(row):
        if abs(row['coefficient']) < 0.01:
            strength = "Minimal"
        elif abs(row['coefficient']) < 0.1:
            strength = "Weak"
        elif abs(row['coefficient']) < 0.5:
            strength = "Moderate"
        elif abs(row['coefficient']) < 1.0:
            strength = "Strong"
        else:
            strength = "Very Strong"
        
        direction = "increases" if row['coefficient'] > 0 else "decreases"
        
        # Odds ratio interpretation
        if row['odds_ratio'] > 1:
            or_interp = f"{row['odds_ratio']:.2f}x more likely"
        else:
            or_interp = f"{1/row['odds_ratio']:.2f}x less likely"
        
        return pd.Series({
            'strength': strength,
            'direction': direction,
            'interpretation': f"{direction.title()} subscription probability ({strength.lower()} effect)",
            'odds_ratio_interp': or_interp
        })
    
    interpretations = coef_df.apply(interpret_coefficient, axis=1)
    coef_df = pd.concat([coef_df, interpretations], axis=1)
    
    print(f"✅ Extracted {len(coef_df)} coefficients")
    print(f"   Intercept: {intercept:.4f}")
    print(f"   Top 5 features by absolute coefficient:")
    for idx, row in coef_df.head(5).iterrows():
        print(f"     - {row['feature']}: {row['coefficient']:.4f} (OR: {row['odds_ratio']:.3f})")
    
    return coef_df, intercept


def analyze_threshold_roi():
    """Analyze threshold optimization results."""
    print("\n🔄 Analyzing threshold optimization...")
    
    if not THRESHOLD_ROI_PATH.exists():
        print("⚠️  Threshold ROI file not found, skipping...")
        return None
    
    roi_df = pd.read_csv(THRESHOLD_ROI_PATH)
    
    # Filter out invalid rows (where total_contacts = 0)
    roi_df = roi_df[roi_df['total_contacts'] > 0]
    
    # Find optimal thresholds
    max_roi_row = roi_df.loc[roi_df['roi'].idxmax()]
    max_precision_row = roi_df.loc[roi_df['precision'].idxmax()]
    
    # Find threshold for ~10% contacts (if exists)
    target_contacts = len(roi_df) * 100 * 0.10  # Rough estimate
    closest_to_10pct = roi_df.iloc[(roi_df['total_contacts'] - target_contacts).abs().argsort()[:1]]
    
    summary = {
        'max_roi': {
            'threshold': max_roi_row['threshold'],
            'roi': max_roi_row['roi'],
            'roi_pct': max_roi_row['roi_pct'],
            'precision': max_roi_row['precision'],
            'recall': max_roi_row['recall'],
            'total_contacts': int(max_roi_row['total_contacts']),
        },
        'max_precision': {
            'threshold': max_precision_row['threshold'],
            'precision': max_precision_row['precision'],
            'recall': max_precision_row['recall'],
            'roi': max_precision_row['roi'],
        },
    }
    
    print(f"✅ Analyzed {len(roi_df)} thresholds")
    print(f"   Max ROI: threshold={summary['max_roi']['threshold']:.3f}, ROI=${summary['max_roi']['roi']:.2f}")
    
    return summary


def create_summary_table(coef_df, threshold_summary=None):
    """Create comprehensive summary table for metrics."""
    print("\n🔄 Creating summary table...")
    
    # Get top features
    top_10_features = coef_df.head(10)
    
    # Create summary
    summary_rows = []
    
    # Performance metrics
    summary_rows.append({
        'Metric': 'ROC-AUC',
        'Value': EXPECTED_METRICS['roc_auc'],
        'Interpretation': 'Excellent discriminatory ability (91.33%)',
        'Category': 'Performance'
    })
    
    summary_rows.append({
        'Metric': 'PR-AUC',
        'Value': EXPECTED_METRICS['pr_auc'],
        'Interpretation': 'Good precision-recall balance for imbalanced data',
        'Category': 'Performance'
    })
    
    summary_rows.append({
        'Metric': 'Precision@10%',
        'Value': EXPECTED_METRICS['precision_at_10'],
        'Interpretation': '59.4% of top 10% predicted leads actually subscribe',
        'Category': 'Business'
    })
    
    summary_rows.append({
        'Metric': 'Lift@10%',
        'Value': EXPECTED_METRICS['lift_at_10'],
        'Interpretation': '5.08x better than random targeting in top decile',
        'Category': 'Business'
    })
    
    if threshold_summary:
        summary_rows.append({
            'Metric': 'Optimal Threshold (Max ROI)',
            'Value': threshold_summary['max_roi']['threshold'],
            'Interpretation': f"ROI: ${threshold_summary['max_roi']['roi']:.2f} | Precision: {threshold_summary['max_roi']['precision']:.1%}",
            'Category': 'Threshold'
        })
    
    # Top predictors
    for idx, row in top_10_features.iterrows():
        summary_rows.append({
            'Metric': f"Feature: {row['feature']}",
            'Value': row['coefficient'],
            'Interpretation': f"{row['direction'].title()} odds ({row['odds_ratio_interp']})",
            'Category': 'Feature'
        })
    
    summary_df = pd.DataFrame(summary_rows)
    print(f"✅ Created summary table with {len(summary_rows)} rows")
    
    return summary_df


def create_slide_summary(coef_df, threshold_summary, summary_table):
    """Create formatted text summary for presentation slide."""
    print("\n🔄 Creating slide summary...")
    
    top_features = coef_df.head(10)
    
    summary_text = f"""
{'='*80}
LOGISTIC REGRESSION RESPONSE ANALYSIS - SLIDE SUMMARY
{'='*80}

1. MODEL PERFORMANCE METRICS
{'-'*80}
   • ROC-AUC: {EXPECTED_METRICS['roc_auc']:.4f}
     → Excellent discriminatory ability (91.33% accuracy in distinguishing subscribers)
   
   • PR-AUC: {EXPECTED_METRICS['pr_auc']:.4f}
     → Good precision-recall balance for imbalanced classification problem
   
   • Precision@10%: {EXPECTED_METRICS['precision_at_10']:.4f}
     → Of the top 10% of predicted leads, 59.4% actually subscribe
   
   • Lift@10%: {EXPECTED_METRICS['lift_at_10']:.2f}x
     → Model is {EXPECTED_METRICS['lift_at_10']:.2f} times better than random targeting in top decile

{'='*80}

2. TOP 10 PREDICTORS (by coefficient magnitude)
{'-'*80}
"""
    
    for idx, (_, row) in enumerate(top_features.iterrows(), 1):
        direction_arrow = "↑" if row['coefficient'] > 0 else "↓"
        summary_text += f"""
   {idx:2d}. {row['feature']:<30} {direction_arrow}
       Coefficient: {row['coefficient']:>8.4f} | Odds Ratio: {row['odds_ratio']:>6.3f}
       Interpretation: {row['interpretation']}
       → {row['odds_ratio_interp']} to subscribe
"""
    
    if threshold_summary:
        summary_text += f"""
{'='*80}

3. THRESHOLD OPTIMIZATION INSIGHTS
{'-'*80}
   • Optimal Threshold (Max ROI): {threshold_summary['max_roi']['threshold']:.3f}
     → ROI: ${threshold_summary['max_roi']['roi']:.2f} ({threshold_summary['max_roi']['roi_pct']:.1f}%)
     → Precision: {threshold_summary['max_roi']['precision']:.1%} | Recall: {threshold_summary['max_roi']['recall']:.1%}
     → Total Contacts: {threshold_summary['max_roi']['total_contacts']:,}
   
   • Max Precision Threshold: {threshold_summary['max_precision']['threshold']:.3f}
     → Precision: {threshold_summary['max_precision']['precision']:.1%}

{'='*80}

4. BUSINESS INSIGHTS
{'-'*80}
   • Targeting Efficiency: Top 10% of leads capture significantly more subscriptions
     than random targeting (5.08x improvement)
   
   • Conversion Rate: At optimal threshold, {threshold_summary['max_roi']['precision']:.1%} of 
     contacted leads will subscribe, yielding positive ROI
   
   • Key Drivers: The model identifies {len([r for r in top_features.head(5).itertuples() if r.coefficient > 0])} strong positive 
     predictors and {len([r for r in top_features.head(5).itertuples() if r.coefficient < 0])} strong negative predictors
     in the top 5 features
"""
    else:
        summary_text += f"""
{'='*80}

4. BUSINESS INSIGHTS
{'-'*80}
   • Targeting Efficiency: Top 10% of leads capture significantly more subscriptions
     than random targeting (5.08x improvement)
   
   • Conversion Rate: At top 10% threshold, 59.4% of contacted leads will subscribe
   
   • Key Drivers: The model identifies {len([r for r in top_features.head(5).itertuples() if r.coefficient > 0])} strong positive 
     predictors and {len([r for r in top_features.head(5).itertuples() if r.coefficient < 0])} strong negative predictors
     in the top 5 features
"""
    
    summary_text += f"""
{'='*80}

5. MODEL CHARACTERISTICS
{'-'*80}
   • Method: Logistic Regression with Elastic Net regularization
   • Regularization: L1/L2 mix (l1_ratio=0.5) to prevent overfitting
   • Class Balancing: Balanced class weights to handle imbalanced data
   • Calibration: Isotonic calibration applied for better probability estimates
   • Features: {len(coef_df)} features (after preprocessing)

{'='*80}

SLIDE RECOMMENDATIONS:
{'-'*80}
   1. Include ROC Curve or Gains Chart visualization (v2_lr_roc.png or v2_lr_gains.png)
   2. Show top 5-7 features with their coefficients/odds ratios
   3. Highlight key metrics: ROC-AUC (0.9133), Lift@10% (5.08x), Precision@10% (59.4%)
   4. Include business insight: "5.08x improvement over random targeting"

{'='*80}
"""
    
    return summary_text


def plot_top_features(coef_df, top_n=15):
    """Create visualization of top features by coefficient."""
    print("\n🔄 Creating feature importance visualization...")
    
    top_features = coef_df.head(top_n).copy()
    top_features = top_features.sort_values('coefficient', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in top_features['coefficient']]
    
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_features['coefficient'], color=colors, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'], fontsize=10)
    ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Logistic Regression: Top {top_n} Features by Coefficient\n(Green = Positive, Red = Negative)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_features.iterrows()):
        value = row['coefficient']
        ax.text(value + (0.01 if value > 0 else -0.01), i, 
                f'{value:.3f}\n(OR: {row["odds_ratio"]:.2f})',
                va='center', ha='left' if value > 0 else 'right',
                fontsize=8)
    
    plt.tight_layout()
    
    output_path = OUTPUT_PATH / "v2_lr_top_features.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualization: {output_path}")
    return output_path


def main():
    """Main execution function."""
    print("="*80)
    print("LOGISTIC REGRESSION SUMMARY GENERATOR")
    print("="*80)
    
    try:
        # Ensure output directory exists
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        
        # Load model and data
        model, feature_names = load_model_and_data()
        
        # Extract coefficients
        coef_df, intercept = extract_coefficients(model, feature_names)
        
        # Save coefficients
        coef_output_path = OUTPUT_PATH / "v2_lr_coefficients.csv"
        coef_df.to_csv(coef_output_path, index=False)
        print(f"✅ Saved coefficients: {coef_output_path}")
        
        # Analyze threshold optimization
        threshold_summary = analyze_threshold_roi()
        
        # Create summary table
        summary_table = create_summary_table(coef_df, threshold_summary)
        summary_output_path = OUTPUT_PATH / "v2_lr_summary_table.csv"
        summary_table.to_csv(summary_output_path, index=False)
        print(f"✅ Saved summary table: {summary_output_path}")
        
        # Create slide summary
        slide_summary = create_slide_summary(coef_df, threshold_summary, summary_table)
        slide_output_path = OUTPUT_PATH / "v2_lr_slide_summary.txt"
        with open(slide_output_path, 'w') as f:
            f.write(slide_summary)
        print(f"✅ Saved slide summary: {slide_output_path}")
        
        # Create visualization
        plot_top_features(coef_df, top_n=15)
        
        # Print summary to console
        print("\n" + slide_summary)
        
        print("\n" + "="*80)
        print("SUMMARY GENERATION COMPLETE")
        print("="*80)
        print(f"\nAll outputs saved in: {OUTPUT_PATH}")
        print(f"  - v2_lr_coefficients.csv")
        print(f"  - v2_lr_summary_table.csv")
        print(f"  - v2_lr_slide_summary.txt")
        print(f"  - v2_lr_top_features.png")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

