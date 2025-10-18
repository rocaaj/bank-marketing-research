"""
response_analysis.py
-----------------------------------
Implements Logistic Regression and Decision Tree models 
for the Bank Marketing dataset using sklearn Pipelines.

Workflow:
1. Load preprocessed dataset from /data
2. Split into train/test sets
3. Build pipelines for Logistic Regression and Decision Tree
4. Train, evaluate, and export key results
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import statsmodels.api as sm
import sys
from datetime import datetime
import io

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = Path("data") / "bank-full-processed.csv"
OUTPUT_PATH = Path("output")
TARGET_COL = "y"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5


# -------------------------
# LOGGING SETUP
# -------------------------
class TeeOutput:
    """Class to capture both console output and write to file simultaneously."""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w')
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()


# -------------------------
# CORE FUNCTIONS
# -------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Load preprocessed dataset from CSV."""
    print(f"\n🔄 Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_data(df: pd.DataFrame):
    """Split dataset into train/test sets."""
    print(f"\n🔄 Splitting data (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"✅ Data split complete - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"📊 Target distribution in training set: {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, name: str):
    """Compute predictive performance metrics."""
    print(f"\n🔄 Evaluating {name} model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n===== {name} Predictive Performance =====")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    # Save confusion matrix plot
    plot_filename = f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(OUTPUT_PATH / plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved confusion matrix plot: {plot_filename}")

    return acc


def cross_validate(model, X, y, model_name):
    """Compute cross-validation scores."""
    print(f"\n🔄 Running {CV_FOLDS}-fold cross-validation for {model_name}...")
    cv_scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring="accuracy")
    print(f"✅ {model_name} Cross-Validation ({CV_FOLDS}-fold):")
    print(f"Mean Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    return cv_scores.mean(), cv_scores.std()


# -------------------------
# LOGISTIC REGRESSION (STATS + SHAP)
# -------------------------
def logistic_regression_analysis(X_train, X_test, y_train, y_test):
    """Train Logistic Regression, compute p-values, odds ratios, SHAP."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    pipe.fit(X_train, y_train)
    evaluate_model(pipe, X_test, y_test, "Logistic Regression")

    # --- Statistical Significance (statsmodels) ---
    # Standardize features as in pipeline
    X_train_scaled = pipe.named_steps["scaler"].transform(X_train)
    X_train_scaled = sm.add_constant(X_train_scaled)
    
    # Target variable is already encoded as 1/0 from preprocessing
    sm_model = sm.Logit(y_train, X_train_scaled)
    sm_results = sm_model.fit(disp=False)

    coef_table = pd.DataFrame({
        "Feature": ["Intercept"] + list(X_train.columns),
        "Coefficient": sm_results.params,
        "StdErr": sm_results.bse,
        "z-value": sm_results.tvalues,
        "p-value": sm_results.pvalues,
        "OddsRatio": np.exp(sm_results.params)
    }).sort_values(by="p-value")

    print("\nLogistic Regression Coefficient Summary (p-values):")
    print(coef_table.head(15).to_string(index=False))

    # --- SHAP Interpretability ---
    explainer = shap.Explainer(pipe.named_steps["model"], pipe.named_steps["scaler"].transform(X_train))
    shap_values = explainer(pipe.named_steps["scaler"].transform(X_test))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary - Logistic Regression")
    
    # Save SHAP plot
    plt.savefig(OUTPUT_PATH / "logistic_regression_shap_summary.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved SHAP summary plot: logistic_regression_shap_summary.png")

    return pipe, coef_table


# -------------------------
# DECISION TREE (SHAP + FEATURE IMPORTANCE)
# -------------------------
def decision_tree_analysis(X_train, X_test, y_train, y_test):
    """Train Decision Tree and compute feature importance + SHAP."""
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # avoid centering for sparse data
        ("model", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))
    ])
    pipe.fit(X_train, y_train)
    evaluate_model(pipe, X_test, y_test, "Decision Tree")

    # --- Feature importance ---
    importances = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": pipe.named_steps["model"].feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nDecision Tree Top Features:")
    print(importances.head(10).to_string(index=False))

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=importances.head(10))
    plt.title("Top Decision Tree Feature Importances")
    plt.tight_layout()
    
    # Save feature importance plot
    plt.savefig(OUTPUT_PATH / "decision_tree_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved feature importance plot: decision_tree_feature_importance.png")

    # --- SHAP values for interpretability ---
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    shap_values = explainer(pipe.named_steps["scaler"].transform(X_test))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary - Decision Tree")
    
    # Save SHAP plot
    plt.savefig(OUTPUT_PATH / "decision_tree_shap_summary.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved SHAP summary plot: decision_tree_shap_summary.png")

    return pipe, importances


# -------------------------
# MAIN
# -------------------------
def main():
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = OUTPUT_PATH / f"response_analysis_log_{timestamp}.txt"
    
    # Create output directory if it doesn't exist
    OUTPUT_PATH.mkdir(exist_ok=True)
    
    # Redirect output to both console and file
    tee_output = TeeOutput(log_filename)
    sys.stdout = tee_output
    
    try:
        print(f"=== Bank Marketing Response Analysis ===")
        print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {log_filename}")
        print(f"{'='*50}\n")
        
        df = load_data(DATA_PATH)
        X_train, X_test, y_train, y_test = split_data(df)

        # Logistic Regression
        log_pipe, log_coefs = logistic_regression_analysis(X_train, X_test, y_train, y_test)
        cross_validate(log_pipe, X_train, y_train, "Logistic Regression")

        # Decision Tree
        tree_pipe, tree_importances = decision_tree_analysis(X_train, X_test, y_train, y_test)
        cross_validate(tree_pipe, X_train, y_train, "Decision Tree")

        # Save coefficient and importance summaries
        log_coefs.to_csv(OUTPUT_PATH / "logistic_significance.csv", index=False)
        tree_importances.to_csv(OUTPUT_PATH / "tree_importances.csv", index=False)
        
        print(f"\nAll results saved in {OUTPUT_PATH.resolve()}")
        print("Saved plots:")
        print("  - logistic_regression_confusion_matrix.png")
        print("  - decision_tree_confusion_matrix.png") 
        print("  - logistic_regression_shap_summary.png")
        print("  - decision_tree_shap_summary.png")
        print("  - decision_tree_feature_importance.png")
        print("Saved data:")
        print("  - logistic_significance.csv")
        print("  - tree_importances.csv")
        print(f"  - response_analysis_log_{timestamp}.txt")
        
        print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        # Restore original stdout and close log file
        sys.stdout = tee_output.terminal
        tee_output.close()


if __name__ == "__main__":
    main()