"""
create_roc_comparison.py
-----------------------------------
Create a combined ROC curve comparison plot for all models.

Generates a single visualization showing:
- LightGBM ROC curve
- XGBoost ROC curve
- CatBoost ROC curve
- Logistic Regression ROC curve
- Random baseline (diagonal)

Outputs: v2_all_models_roc_comparison.png
"""

import sys
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "bank-full-processed-v2.csv"
OUTPUT_PATH = BASE_DIR / "output" / "classification"
MODEL_DIR = OUTPUT_PATH / "models"
TARGET_COL = "y"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model files
MODEL_FILES = {
    'Logistic Regression': MODEL_DIR / "v2_logistic_model.pkl",
    'LightGBM': MODEL_DIR / "v2_lightgbm_model.pkl",
    'XGBoost': MODEL_DIR / "v2_xgboost_model.pkl",
    'CatBoost': MODEL_DIR / "v2_catboost_model.pkl",
}

# Colors for each model
MODEL_COLORS = {
    'Logistic Regression': '#e74c3c',  # Red
    'LightGBM': '#3498db',              # Blue
    'XGBoost': '#2ecc71',               # Green
    'CatBoost': '#f39c12',              # Orange
}

# Line styles
MODEL_STYLES = {
    'Logistic Regression': '-',
    'LightGBM': '-',
    'XGBoost': '-',
    'CatBoost': '-',
}


def load_data():
    """Load the preprocessed dataset."""
    print("🔄 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Use same train-test split as in training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"✅ Loaded data - Test set: {X_test.shape[0]} samples")
    return X_test, y_test


def load_model(model_path):
    """Load a saved model from pickle file."""
    if not model_path.exists():
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"⚠️  Error loading {model_path}: {e}")
        return None


def predict_proba_safe(model, X):
    """Safely predict probabilities from any model type."""
    try:
        # Handle CalibratedClassifierCV
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]  # Positive class probability
        
        # Handle Pipeline
        if hasattr(model, 'named_steps'):
            return model.predict_proba(X)[:, 1]
        
        return model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"⚠️  Error in prediction: {e}")
        return None


def compute_roc_curve(y_true, y_scores):
    """Compute ROC curve and AUC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)
    return fpr, tpr, auc_score


def create_comparison_plot(roc_data):
    """Create a combined ROC curve comparison plot."""
    print("\n🔄 Creating ROC curve comparison plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot random baseline (diagonal)
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', 
            linewidth=1.5, alpha=0.7)
    
    # Plot each model's ROC curve
    for model_name, (fpr, tpr, auc_score) in sorted(roc_data.items(), 
                                                      key=lambda x: x[1][2], 
                                                      reverse=True):
        color = MODEL_COLORS.get(model_name, '#333333')
        style = MODEL_STYLES.get(model_name, '-')
        ax.plot(fpr, tpr, 
                label=f'{model_name} (AUC = {auc_score:.4f})',
                linewidth=2.5,
                linestyle=style,
                color=color,
                alpha=0.8)
    
    # Formatting
    ax.set_xlabel('False Positive Rate (1 - Specificity)', 
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', 
                  fontsize=13, fontweight='bold')
    ax.set_title('ROC Curve Comparison: All Models', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_PATH / "v2_all_models_roc_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comparison plot: {output_path}")
    return output_path


def print_comparison_summary(roc_data):
    """Print a summary table of ROC-AUC scores."""
    print("\n" + "=" * 70)
    print("ROC-AUC COMPARISON SUMMARY")
    print("=" * 70)
    
    # Sort by AUC descending
    sorted_models = sorted(roc_data.items(), key=lambda x: x[1][2], reverse=True)
    
    print(f"\n{'Rank':<6} {'Model':<25} {'ROC-AUC':<12} {'Difference from Best':<20}")
    print("-" * 70)
    
    best_auc = sorted_models[0][1][2]  # Best AUC
    
    for rank, (model_name, (_, _, auc_score)) in enumerate(sorted_models, 1):
        diff = best_auc - auc_score
        diff_str = f"-{diff:.4f}" if diff > 0 else "0.0000 (Best)"
        print(f"{rank:<6} {model_name:<25} {auc_score:<12.4f} {diff_str:<20}")
    
    print("=" * 70)


def main():
    """Main execution function."""
    print("=" * 70)
    print("ROC CURVE COMPARISON GENERATOR")
    print("=" * 70)
    
    # Ensure output directory exists
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    X_test, y_test = load_data()
    
    # Load models and compute ROC curves
    roc_data = {}
    models_loaded = 0
    
    print("\n🔄 Loading models and computing ROC curves...")
    
    for model_name, model_path in MODEL_FILES.items():
        print(f"\n  Processing {model_name}...")
        
        model = load_model(model_path)
        if model is None:
            print(f"    ⚠️  Model file not found or failed to load: {model_path.name}")
            continue
        
        y_scores = predict_proba_safe(model, X_test)
        if y_scores is None:
            print(f"    ⚠️  Failed to generate predictions for {model_name}")
            continue
        
        fpr, tpr, auc_score = compute_roc_curve(y_test, y_scores)
        roc_data[model_name] = (fpr, tpr, auc_score)
        
        print(f"    ✅ {model_name}: AUC = {auc_score:.4f}")
        models_loaded += 1
    
    if models_loaded == 0:
        print("\n❌ No models were successfully loaded. Exiting.")
        sys.exit(1)
    
    print(f"\n✅ Successfully loaded {models_loaded} model(s)")
    
    # Print summary
    print_comparison_summary(roc_data)
    
    # Create comparison plot
    if len(roc_data) > 0:
        output_path = create_comparison_plot(roc_data)
        
        print("\n" + "=" * 70)
        print("COMPARISON PLOT GENERATION COMPLETE")
        print("=" * 70)
        print(f"\nOutput saved to: {output_path}")
        
        # Additional insights
        if len(roc_data) >= 2:
            sorted_aucs = sorted([(name, auc) for name, (_, _, auc) in roc_data.items()], 
                               key=lambda x: x[1], reverse=True)
            best_name, best_auc = sorted_aucs[0]
            worst_name, worst_auc = sorted_aucs[-1]
            improvement = ((best_auc - worst_auc) / worst_auc) * 100
            
            print(f"\n📊 Key Insights:")
            print(f"   • Best model: {best_name} (AUC = {best_auc:.4f})")
            print(f"   • Performance range: {worst_auc:.4f} - {best_auc:.4f}")
            print(f"   • Improvement: {improvement:.2f}% over lowest performer")
    
    else:
        print("\n⚠️  No ROC data available for plotting.")


if __name__ == "__main__":
    main()

