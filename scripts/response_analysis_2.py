"""
response_analysis_2.py
-----------------------------------
Supervised lead scoring with state-of-the-art classifiers for the Bank Marketing dataset.

Models:
- Logistic Regression (Elastic Net)
- LightGBM (with hyperparameter tuning)
- XGBoost (with hyperparameter tuning)
- CatBoost (with hyperparameter tuning)

Features:
- Hyperparameter tuning (Optuna)
- Probability calibration (Isotonic/Platt)
- Threshold optimization & ROI analysis
- Business metrics (Precision@K, Lift@K, Gains charts)
- SHAP explainability
- Model persistence for reproducibility

Outputs saved under: output/classification (prefixed with v2_*)
"""

import sys
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, brier_score_loss, 
    roc_curve, precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Optional libraries
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# -------------------------
# CONFIG
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "bank-full-processed-v2.csv"
OUTPUT_PATH = BASE_DIR / "output"
OUTPUT_CLASSIFICATION = OUTPUT_PATH / "classification"
TARGET_COL = "y"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5
N_TRIALS = 50  # Optuna trials per model

# Feature flags
ENABLE_TUNING = False  # Set to True to enable hyperparameter tuning (slower but better results)

# Business parameters (adjustable)
CONTACT_COST = 1.0  # Cost per contact in dollars
SUBSCRIPTION_REVENUE = 100.0  # Revenue per subscription in dollars


class TeeOutput:
    """Redirect output to both console and file."""
    def __init__(self, file_path: Path):
        self.terminal = sys.stdout
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
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
# DATA UTILS
# -------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Load preprocessed dataset."""
    print(f"\n🔄 Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def stratified_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train-test split."""
    print(f"\n🔄 Stratified split (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"✅ Split complete - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"📊 Train target distribution: {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


def create_cv_folds() -> StratifiedKFold:
    """Create stratified K-fold cross-validator."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


# -------------------------
# METRICS / EVALUATION
# -------------------------
def compute_precision_at_k(y_true: pd.Series, y_scores: np.ndarray, k: float = 0.1) -> float:
    """Compute precision at top K% of predictions."""
    n = len(y_true)
    top_n = max(1, int(n * k))
    idx = np.argsort(-y_scores)[:top_n]
    return float(y_true.iloc[idx].mean())


def compute_lift_at_k(y_true: pd.Series, y_scores: np.ndarray, k: float = 0.1) -> float:
    """Compute lift at top K% of predictions."""
    precision_k = compute_precision_at_k(y_true, y_scores, k)
    baseline_rate = float(y_true.mean())
    return precision_k / baseline_rate if baseline_rate > 0 else np.nan


def compute_gains_data(y_true: pd.Series, y_scores: np.ndarray, n_bins: int = 20) -> Dict[str, np.ndarray]:
    """Compute cumulative gains data for gains chart."""
    sorted_idx = np.argsort(-y_scores)
    sorted_true = y_true.iloc[sorted_idx].values
    n = len(sorted_true)
    bin_size = n // n_bins
    
    cumulative_gains = []
    cumulative_contacts = []
    
    for i in range(1, n_bins + 1):
        end_idx = i * bin_size
        cumulative_gains.append(sorted_true[:end_idx].sum())
        cumulative_contacts.append(end_idx)
    
    cumulative_gains = np.array(cumulative_gains)
    cumulative_contacts = np.array(cumulative_contacts)
    
    # Normalize to percentages
    total_positive = sorted_true.sum()
    gains_pct = (cumulative_gains / total_positive * 100) if total_positive > 0 else cumulative_gains
    contacts_pct = cumulative_contacts / n * 100
    
    return {
        'contacts_pct': contacts_pct,
        'gains_pct': gains_pct,
        'lift': gains_pct / contacts_pct if contacts_pct[0] > 0 else gains_pct
    }


def evaluate_scores(y_true: pd.Series, y_scores: np.ndarray, name: str, prefix: str) -> Dict[str, Any]:
    """Comprehensive evaluation with business metrics."""
    roc_auc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Business metrics at different thresholds
    p_at_05 = compute_precision_at_k(y_true, y_scores, 0.05)
    p_at_10 = compute_precision_at_k(y_true, y_scores, 0.10)
    p_at_20 = compute_precision_at_k(y_true, y_scores, 0.20)
    lift_at_10 = compute_lift_at_k(y_true, y_scores, 0.10)

    print(f"\n===== {name} - Metrics =====")
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"Precision@5%: {p_at_05:.4f} | Precision@10%: {p_at_10:.4f} | Precision@20%: {p_at_20:.4f}")
    print(f"Lift@10%: {lift_at_10:.2f}x")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{name} - ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_CLASSIFICATION / f"v2_{prefix}_roc.png", dpi=300, bbox_inches='tight')
    plt.close()

    # PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{name} - Precision-Recall Curve (AUC={pr_auc:.3f})', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_CLASSIFICATION / f"v2_{prefix}_pr.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Gains chart
    gains_data = compute_gains_data(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(gains_data['contacts_pct'], gains_data['gains_pct'], marker='o', linewidth=2, label='Model')
    plt.plot([0, 100], [0, 100], 'k--', label='Random', linewidth=1)
    plt.xlabel('% of Contacts', fontsize=12)
    plt.ylabel('% of Subscriptions Captured', fontsize=12)
    plt.title(f'{name} - Cumulative Gains Chart', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_CLASSIFICATION / f"v2_{prefix}_gains.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Lift chart
    plt.figure(figsize=(8, 6))
    plt.plot(gains_data['contacts_pct'], gains_data['lift'], marker='o', linewidth=2)
    plt.axhline(y=1.0, color='k', linestyle='--', label='Baseline', linewidth=1)
    plt.xlabel('% of Contacts', fontsize=12)
    plt.ylabel('Lift', fontsize=12)
    plt.title(f'{name} - Lift Chart', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_CLASSIFICATION / f"v2_{prefix}_lift.png", dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision_at_5': p_at_05,
        'precision_at_10': p_at_10,
        'precision_at_20': p_at_20,
        'lift_at_10': lift_at_10,
        'fpr': fpr,
        'tpr': tpr,
        'gains_data': gains_data,
    }


def plot_calibration(y_true: pd.Series, y_proba: np.ndarray, name: str, prefix: str) -> float:
    """Plot calibration curve and return Brier score."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy='quantile')
    brier = brier_score_loss(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Calibrated predictions')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1)
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Observed Frequency', fontsize=12)
    plt.title(f'{name} - Calibration Curve (Brier Score: {brier:.4f})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_CLASSIFICATION / f"v2_{prefix}_calibration.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return brier


# -------------------------
# HYPERPARAMETER TUNING (Optuna)
# -------------------------
def tune_lightgbm(X_train: pd.DataFrame, y_train: pd.Series) -> Optional[Dict[str, Any]]:
    """Tune LightGBM hyperparameters using Optuna."""
    if not ENABLE_TUNING:
        print(f"\n⏭️  Skipping LightGBM tuning (ENABLE_TUNING=False). Using default parameters.")
        return None
    
    if not LGB_AVAILABLE or not OPTUNA_AVAILABLE:
        print(f"\n⚠️  LightGBM tuning unavailable (library missing). Using default parameters.")
        return None
    
    print(f"\n🔧 Tuning LightGBM with Optuna ({N_TRIALS} trials)...")
    cv = create_cv_folds()
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary',
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbose': -1,
        }
        
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_pred_proba))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params.copy()
    best_params.update({
        'scale_pos_weight': scale_pos_weight,
        'objective': 'binary',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
    })
    
    print(f"✅ Best LightGBM CV ROC-AUC: {study.best_value:.4f}")
    return best_params


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> Optional[Dict[str, Any]]:
    """Tune XGBoost hyperparameters using Optuna."""
    if not ENABLE_TUNING:
        print(f"\n⏭️  Skipping XGBoost tuning (ENABLE_TUNING=False). Using default parameters.")
        return None
    
    if not XGB_AVAILABLE or not OPTUNA_AVAILABLE:
        print(f"\n⚠️  XGBoost tuning unavailable (library missing). Using default parameters.")
        return None
    
    print(f"\n🔧 Tuning XGBoost with Optuna ({N_TRIALS} trials)...")
    cv = create_cv_folds()
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary:logistic',
            'random_state': RANDOM_STATE,
            'tree_method': 'hist',
            'eval_metric': 'auc',
            'n_jobs': -1,
        }
        
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, verbose=False)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_pred_proba))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params.copy()
    best_params.update({
        'scale_pos_weight': scale_pos_weight,
        'objective': 'binary:logistic',
        'random_state': RANDOM_STATE,
        'tree_method': 'hist',
        'eval_metric': 'auc',
        'n_jobs': -1,
    })
    
    print(f"✅ Best XGBoost CV ROC-AUC: {study.best_value:.4f}")
    return best_params


def tune_catboost(X_train: pd.DataFrame, y_train: pd.Series) -> Optional[Dict[str, Any]]:
    """Tune CatBoost hyperparameters using Optuna."""
    if not ENABLE_TUNING:
        print(f"\n⏭️  Skipping CatBoost tuning (ENABLE_TUNING=False). Using default parameters.")
        return None
    
    if not CAT_AVAILABLE or not OPTUNA_AVAILABLE:
        print(f"\n⚠️  CatBoost tuning unavailable (library missing). Using default parameters.")
        return None
    
    print(f"\n🔧 Tuning CatBoost with Optuna ({N_TRIALS} trials)...")
    cv = create_cv_folds()
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'scale_pos_weight': scale_pos_weight,
            'loss_function': 'Logloss',
            'random_seed': RANDOM_STATE,
            'verbose': False,
        }
        
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_pred_proba))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params.copy()
    best_params.update({
        'scale_pos_weight': scale_pos_weight,
        'loss_function': 'Logloss',
        'random_seed': RANDOM_STATE,
        'verbose': False,
    })
    
    print(f"✅ Best CatBoost CV ROC-AUC: {study.best_value:.4f}")
    return best_params


# -------------------------
# MODEL TRAINING
# -------------------------
def train_logistic(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train Logistic Regression with Elastic Net."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty='elasticnet', solver='saga', max_iter=5000,
            l1_ratio=0.5, C=1.0, class_weight='balanced', 
            random_state=RANDOM_STATE
        ))
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, 
                   params: Optional[Dict] = None) -> Optional[lgb.LGBMClassifier]:
    """Train LightGBM with optional tuned parameters."""
    if not LGB_AVAILABLE:
        return None
    
    if params is None:
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0
        params = {
            'n_estimators': 600,
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': -1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary',
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbose': -1,
        }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  params: Optional[Dict] = None) -> Optional[xgb.XGBClassifier]:
    """Train XGBoost with optional tuned parameters."""
    if not XGB_AVAILABLE:
        return None
    
    if params is None:
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0
        params = {
            'n_estimators': 800,
            'learning_rate': 0.03,
            'max_depth': 5,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary:logistic',
            'random_state': RANDOM_STATE,
            'tree_method': 'hist',
            'eval_metric': 'auc',
            'n_jobs': -1,
        }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    return model


def train_catboost(X_train: pd.DataFrame, y_train: pd.Series,
                   params: Optional[Dict] = None) -> Optional[CatBoostClassifier]:
    """Train CatBoost with optional tuned parameters."""
    if not CAT_AVAILABLE:
        return None
    
    if params is None:
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0
        params = {
            'iterations': 800,
            'learning_rate': 0.03,
            'depth': 6,
            'scale_pos_weight': scale_pos_weight,
            'loss_function': 'Logloss',
            'random_seed': RANDOM_STATE,
            'verbose': False,
        }
    
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    return model


def predict_proba_generic(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Generic predict_proba wrapper."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    raise AttributeError(f'Model {type(model)} does not support predict_proba')


# -------------------------
# CALIBRATION
# -------------------------
def calibrate_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series, 
                    method: str = 'isotonic') -> CalibratedClassifierCV:
    """Calibrate model probabilities."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    calibrated = CalibratedClassifierCV(model, method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated


# -------------------------
# THRESHOLD OPTIMIZATION & ROI
# -------------------------
def compute_roi_at_threshold(y_true: pd.Series, y_proba: np.ndarray, 
                              threshold: float) -> Dict[str, float]:
    """Compute ROI metrics at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    total_contacts = tp + fp
    
    revenue = tp * SUBSCRIPTION_REVENUE
    cost = total_contacts * CONTACT_COST
    roi = revenue - cost
    roi_pct = (roi / cost * 100) if cost > 0 else 0.0
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'total_contacts': total_contacts,
        'true_positives': tp,
        'false_positives': fp,
        'revenue': revenue,
        'cost': cost,
        'roi': roi,
        'roi_pct': roi_pct,
    }


def optimize_thresholds(y_true: pd.Series, y_proba: np.ndarray, 
                        n_thresholds: int = 50) -> pd.DataFrame:
    """Optimize thresholds for different business objectives."""
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    results = []
    
    for thresh in thresholds:
        metrics = compute_roi_at_threshold(y_true, y_proba, thresh)
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Find optimal thresholds
    max_roi_thresh = df.loc[df['roi'].idxmax(), 'threshold']
    max_precision_thresh = df.loc[df['precision'].idxmax(), 'threshold']
    p_at_10_thresh = None  # Find threshold that gives ~10% contacts
    target_contacts = len(y_true) * 0.10
    closest_idx = (df['total_contacts'] - target_contacts).abs().idxmin()
    p_at_10_thresh = df.loc[closest_idx, 'threshold']
    
    print(f"\n📊 Threshold Optimization Summary:")
    print(f"   Max ROI threshold: {max_roi_thresh:.3f} (ROI: ${df.loc[df['roi'].idxmax(), 'roi']:.2f})")
    print(f"   Max Precision threshold: {max_precision_thresh:.3f} (Precision: {df.loc[df['precision'].idxmax(), 'precision']:.3f})")
    print(f"   ~10% contacts threshold: {p_at_10_thresh:.3f}")
    
    # Plot ROI curve
    plt.figure(figsize=(10, 6))
    plt.plot(df['threshold'], df['roi'], label='ROI', linewidth=2)
    plt.axvline(max_roi_thresh, color='r', linestyle='--', label=f'Optimal (t={max_roi_thresh:.3f})')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('ROI ($)', fontsize=12)
    plt.title('ROI by Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_CLASSIFICATION / "v2_threshold_roi.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


# -------------------------
# SHAP EXPLAINABILITY
# -------------------------
def compute_shap_values(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        name: str, prefix: str) -> Optional[Dict[str, Any]]:
    """Compute and visualize SHAP values."""
    if not SHAP_AVAILABLE:
        print(f"⚠️  SHAP not available for {name}")
        return None
    
    try:
        print(f"\n🔍 Computing SHAP values for {name}...")
        
        # Use sample for faster computation
        sample_size = min(500, len(X_test))
        X_test_sample = X_test.iloc[:sample_size].copy()
        X_train_sample = X_train.iloc[:min(1000, len(X_train))].copy()
        
        # Handle Pipeline objects (e.g., Logistic Regression)
        actual_model = model
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            # It's a Pipeline - extract the actual model and scale data
            actual_model = model.named_steps['model']
            if 'scaler' in model.named_steps:
                scaler = model.named_steps['scaler']
                X_train_sample_scaled = pd.DataFrame(
                    scaler.transform(X_train_sample),
                    columns=X_train_sample.columns,
                    index=X_train_sample.index
                )
                X_test_sample_scaled = pd.DataFrame(
                    scaler.transform(X_test_sample),
                    columns=X_test_sample.columns,
                    index=X_test_sample.index
                )
                X_train_sample = X_train_sample_scaled
                X_test_sample = X_test_sample_scaled
        
        # Convert to numpy arrays for SHAP compatibility
        X_train_sample_np = X_train_sample.values
        X_test_sample_np = X_test_sample.values
        
        # Tree models use TreeExplainer, others use KernelExplainer
        if isinstance(actual_model, (lgb.LGBMClassifier, xgb.XGBClassifier, CatBoostClassifier)):
            # Ensure data is numeric and handle any non-numeric issues
            X_test_sample_np = X_test_sample_np.astype(float)
            X_train_sample_np = X_train_sample_np.astype(float)
            
            # TreeExplainer for tree-based models
            if isinstance(actual_model, xgb.XGBClassifier):
                # XGBoost sometimes needs special handling
                try:
                    # Try with model_output='probability' first
                    explainer = shap.TreeExplainer(actual_model, model_output='probability')
                    shap_values = explainer.shap_values(X_test_sample_np)
                except Exception as e1:
                    try:
                        # Fallback: standard TreeExplainer
                        explainer = shap.TreeExplainer(actual_model)
                        shap_values = explainer.shap_values(X_test_sample_np)
                    except Exception as e2:
                        # Last resort: use Explainer (newer SHAP API)
                        try:
                            explainer = shap.Explainer(actual_model, X_train_sample_np)
                            shap_output = explainer(X_test_sample_np)
                            shap_values = shap_output.values
                            if len(shap_values.shape) == 3:
                                shap_values = shap_values[:, :, 1]  # Positive class
                        except Exception as e3:
                            print(f"   All SHAP methods failed for XGBoost: {e3}")
                            raise
            else:
                # LightGBM and CatBoost
                explainer = shap.TreeExplainer(actual_model)
                shap_values = explainer.shap_values(X_test_sample_np)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Binary classification
        else:
            # For non-tree models (Logistic Regression, etc.)
            # Use a wrapper function that works with the model
            def predict_proba_wrapper(X):
                if hasattr(actual_model, 'predict_proba'):
                    return actual_model.predict_proba(X)
                else:
                    # Fallback for pipeline
                    return model.predict_proba(pd.DataFrame(X, columns=X_test_sample.columns))
            
            explainer = shap.KernelExplainer(predict_proba_wrapper, X_train_sample_np)
            shap_values_full = explainer.shap_values(X_test_sample_np)
            # Handle different return formats
            if isinstance(shap_values_full, list) and len(shap_values_full) > 1:
                shap_values = shap_values_full[1]  # Positive class
            elif isinstance(shap_values_full, np.ndarray) and len(shap_values_full.shape) == 3:
                shap_values = shap_values_full[:, :, 1]  # Positive class
            else:
                shap_values = shap_values_full
        
        # Ensure shap_values is 2D numpy array
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        shap_values = np.atleast_2d(shap_values)
        
        # Summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_test_sample_np, 
                         feature_names=X_test_sample.columns.tolist(),
                         show=False, plot_type="bar")
        plt.title(f'{name} - SHAP Feature Importance', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_CLASSIFICATION / f"v2_{prefix}_shap_bar.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': X_test_sample.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(OUTPUT_CLASSIFICATION / f"v2_{prefix}_shap_importance.csv", index=False)
        
        print(f"✅ SHAP analysis complete for {name}")
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'feature_importance': feature_importance,
        }
    
    except Exception as e:
        print(f"⚠️  Error computing SHAP for {name}: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return None


# -------------------------
# MODEL PERSISTENCE
# -------------------------
def save_model(model: Any, model_name: str, params: Optional[Dict] = None):
    """Save model and parameters for reproducibility."""
    model_dir = OUTPUT_CLASSIFICATION / "models"
    model_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = model_dir / f"v2_{model_name}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save parameters
    if params:
        params_path = model_dir / f"v2_{model_name}_params.json"
        # Convert numpy types to Python types for JSON
        params_serializable = {}
        for k, v in params.items():
            if isinstance(v, (np.integer, np.floating)):
                params_serializable[k] = float(v)
            elif isinstance(v, np.ndarray):
                params_serializable[k] = v.tolist()
            else:
                params_serializable[k] = v
        
        with open(params_path, 'w') as f:
            json.dump(params_serializable, f, indent=2)
    
    print(f"💾 Saved model: {model_path}")


# -------------------------
# MAIN PIPELINE
# -------------------------
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = OUTPUT_PATH / f"response_analysis_v2_log_{timestamp}.txt"
    
    tee_output = TeeOutput(log_filename)
    sys.stdout = tee_output
    
    try:
        print("=" * 70)
        print("Bank Marketing Response Analysis v2 - Supervised Lead Scoring")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {log_filename}")
        print(f"Random State: {RANDOM_STATE} | CV Folds: {N_SPLITS}")
        print(f"Hyperparameter Tuning: {'ENABLED' if ENABLE_TUNING else 'DISABLED'} (Trials: {N_TRIALS if ENABLE_TUNING else 'N/A'})")
        print("=" * 70)
        
        # Load data
        df = load_data(DATA_PATH)
        X_train, X_test, y_train, y_test = stratified_split(df)
        
        results = {}
        models = {}
        best_params = {}
        
        # ============================================================
        # 1. LOGISTIC REGRESSION (Baseline)
        # ============================================================
        print("\n" + "=" * 70)
        print("1. LOGISTIC REGRESSION (Elastic Net)")
        print("=" * 70)
        
        log_model = train_logistic(X_train, y_train)
        models['logistic'] = log_model
        y_scores_log = predict_proba_generic(log_model, X_test)
        
        # Calibrate
        log_model_cal = calibrate_model(log_model, X_train, y_train, method='isotonic')
        y_scores_log_cal = predict_proba_generic(log_model_cal, X_test)
        
        res_log = evaluate_scores(y_test, y_scores_log_cal, 'Logistic Regression (EN)', 'lr')
        brier_log = plot_calibration(y_test, y_scores_log_cal, 'Logistic Regression (EN)', 'lr')
        results['Logistic Regression (EN)'] = {**res_log, 'brier': brier_log, 'model': log_model_cal}
        
        # Threshold optimization
        roi_log = optimize_thresholds(y_test, y_scores_log_cal)
        roi_log.to_csv(OUTPUT_CLASSIFICATION / "v2_lr_threshold_roi.csv", index=False)
        
        # SHAP (for interpretability)
        shap_log = compute_shap_values(log_model, X_train, X_test, 'Logistic Regression', 'lr')
        
        save_model(log_model_cal, 'logistic')
        
        # ============================================================
        # 2. LIGHTGBM (with tuning)
        # ============================================================
        if LGB_AVAILABLE:
            print("\n" + "=" * 70)
            print("2. LIGHTGBM")
            print("=" * 70)
            
            lgb_params = tune_lightgbm(X_train, y_train)
            if lgb_params:
                best_params['lightgbm'] = lgb_params
                lgb_model = train_lightgbm(X_train, y_train, lgb_params)
            else:
                lgb_model = train_lightgbm(X_train, y_train)
            
            models['lightgbm'] = lgb_model
            y_scores_lgb = predict_proba_generic(lgb_model, X_test)
            
            # Calibrate
            lgb_model_cal = calibrate_model(lgb_model, X_train, y_train, method='isotonic')
            y_scores_lgb_cal = predict_proba_generic(lgb_model_cal, X_test)
            
            res_lgb = evaluate_scores(y_test, y_scores_lgb_cal, 'LightGBM', 'lgb')
            brier_lgb = plot_calibration(y_test, y_scores_lgb_cal, 'LightGBM', 'lgb')
            results['LightGBM'] = {**res_lgb, 'brier': brier_lgb, 'model': lgb_model_cal}
            
            # Threshold optimization
            roi_lgb = optimize_thresholds(y_test, y_scores_lgb_cal)
            roi_lgb.to_csv(OUTPUT_CLASSIFICATION / "v2_lgb_threshold_roi.csv", index=False)
            
            # SHAP
            shap_lgb = compute_shap_values(lgb_model, X_train, X_test, 'LightGBM', 'lgb')
            
            save_model(lgb_model_cal, 'lightgbm', lgb_params)
        
        # ============================================================
        # 3. XGBOOST (with tuning)
        # ============================================================
        if XGB_AVAILABLE:
            print("\n" + "=" * 70)
            print("3. XGBOOST")
            print("=" * 70)
            
            xgb_params = tune_xgboost(X_train, y_train)
            if xgb_params:
                best_params['xgboost'] = xgb_params
                xgb_model = train_xgboost(X_train, y_train, xgb_params)
            else:
                xgb_model = train_xgboost(X_train, y_train)
            
            models['xgboost'] = xgb_model
            y_scores_xgb = predict_proba_generic(xgb_model, X_test)
            
            # Calibrate
            xgb_model_cal = calibrate_model(xgb_model, X_train, y_train, method='isotonic')
            y_scores_xgb_cal = predict_proba_generic(xgb_model_cal, X_test)
            
            res_xgb = evaluate_scores(y_test, y_scores_xgb_cal, 'XGBoost', 'xgb')
            brier_xgb = plot_calibration(y_test, y_scores_xgb_cal, 'XGBoost', 'xgb')
            results['XGBoost'] = {**res_xgb, 'brier': brier_xgb, 'model': xgb_model_cal}
            
            # Threshold optimization
            roi_xgb = optimize_thresholds(y_test, y_scores_xgb_cal)
            roi_xgb.to_csv(OUTPUT_CLASSIFICATION / "v2_xgb_threshold_roi.csv", index=False)
            
            # SHAP
            shap_xgb = compute_shap_values(xgb_model, X_train, X_test, 'XGBoost', 'xgb')
            
            save_model(xgb_model_cal, 'xgboost', xgb_params)
        
        # ============================================================
        # 4. CATBOOST (with tuning)
        # ============================================================
        if CAT_AVAILABLE:
            print("\n" + "=" * 70)
            print("4. CATBOOST")
            print("=" * 70)
            
            cat_params = tune_catboost(X_train, y_train)
            if cat_params:
                best_params['catboost'] = cat_params
                cat_model = train_catboost(X_train, y_train, cat_params)
            else:
                cat_model = train_catboost(X_train, y_train)
            
            models['catboost'] = cat_model
            y_scores_cat = predict_proba_generic(cat_model, X_test)
            
            # Calibrate
            cat_model_cal = calibrate_model(cat_model, X_train, y_train, method='isotonic')
            y_scores_cat_cal = predict_proba_generic(cat_model_cal, X_test)
            
            res_cat = evaluate_scores(y_test, y_scores_cat_cal, 'CatBoost', 'cat')
            brier_cat = plot_calibration(y_test, y_scores_cat_cal, 'CatBoost', 'cat')
            results['CatBoost'] = {**res_cat, 'brier': brier_cat, 'model': cat_model_cal}
            
            # Threshold optimization
            roi_cat = optimize_thresholds(y_test, y_scores_cat_cal)
            roi_cat.to_csv(OUTPUT_CLASSIFICATION / "v2_cat_threshold_roi.csv", index=False)
            
            # SHAP
            shap_cat = compute_shap_values(cat_model, X_train, X_test, 'CatBoost', 'cat')
            
            save_model(cat_model_cal, 'catboost', cat_params)
        
        # ============================================================
        # 5. COMPARISON & SUMMARY
        # ============================================================
        print("\n" + "=" * 70)
        print("5. MODEL COMPARISON & SUMMARY")
        print("=" * 70)
        
        # Create comparison table
        comp_rows = []
        for name, r in results.items():
            comp_rows.append({
                'Model': name,
                'ROC-AUC': r.get('roc_auc', np.nan),
                'PR-AUC': r.get('pr_auc', np.nan),
                'Precision@5%': r.get('precision_at_5', np.nan),
                'Precision@10%': r.get('precision_at_10', np.nan),
                'Precision@20%': r.get('precision_at_20', np.nan),
                'Lift@10%': r.get('lift_at_10', np.nan),
                'Brier': r.get('brier', np.nan),
            })
        
        df_comp = pd.DataFrame(comp_rows).sort_values('ROC-AUC', ascending=False)
        print("\n" + "=" * 70)
        print("MODEL COMPARISON (sorted by ROC-AUC)")
        print("=" * 70)
        print(df_comp.to_string(index=False))
        df_comp.to_csv(OUTPUT_CLASSIFICATION / "v2_model_comparison.csv", index=False)
        
        # Save configuration
        config = {
            'random_state': RANDOM_STATE,
            'test_size': TEST_SIZE,
            'n_splits': N_SPLITS,
            'n_trials': N_TRIALS,
            'contact_cost': CONTACT_COST,
            'subscription_revenue': SUBSCRIPTION_REVENUE,
            'best_params': best_params,
        }
        with open(OUTPUT_CLASSIFICATION / "v2_config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Final summary
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nAll artifacts saved under: {OUTPUT_CLASSIFICATION}")
        print(f"\nModels saved in: {OUTPUT_CLASSIFICATION / 'models'}")
        print(f"\nBest model: {df_comp.iloc[0]['Model']} (ROC-AUC: {df_comp.iloc[0]['ROC-AUC']:.4f})")
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        sys.stdout = tee_output.terminal
        tee_output.close()


if __name__ == "__main__":
    main()
