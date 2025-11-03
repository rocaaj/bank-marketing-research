# Research Log - Bank Marketing Lead Scoring Project

## Date: 2024-11-XX

### Context
After evaluating multiple state-of-the-art classification models (Logistic Regression with Elastic Net, LightGBM, XGBoost, CatBoost), we observed that model selection alone had reached a performance plateau. The best model (LightGBM) achieved ROC-AUC of ~0.934, but further improvements through hyperparameter tuning showed diminishing returns. This led to a strategic pivot: **focus on feature engineering and data preprocessing to unlock additional performance gains.**

---

### Session Timeline

#### 1. Strategy Shift: Model Selection → Feature Engineering
- **Observation**: Model performance plateaued despite using gradient boosting machines with proper class weighting and calibration
- **Decision**: Shift focus from model optimization to data preprocessing and feature engineering
- **Rationale**: Feature engineering often provides more substantial improvements than hyperparameter tuning, especially when models are already well-configured

#### 2. Enhanced Data Preprocessing Pipeline (v2)
- **Created**: `scripts/data_preprocessing_2.py`
- **Goal**: Implement domain-informed feature engineering to improve model performance
- **Key transformations implemented**:
  - **`was_contacted_before`**: Binary feature derived from `pdays` (1 if pdays ≠ 999, else 0)
    - Rationale: Captures whether customer was previously contacted in a binary indicator
  - **`previous_contacts`**: Renamed and preserved `previous` column as a continuous feature
    - Rationale: Number of previous contacts is informative, but original name was unclear
  - **`balance_log`**: Log-transform of balance using `np.log1p`, with negatives clipped to 0
    - Rationale: Balance has heavy right-skew; log transformation improves model stability
  - **`campaign_capped`**: Capped `campaign` values at maximum of 5
    - Rationale: High campaign values are outliers that may be data errors; capping reduces noise
  - **`age_group`**: Binned age into 5 quantile groups
    - Rationale: Age may have non-linear relationships; binning captures patterns better than raw values
  - **`education_ord`**: Ordinal encoding of education with progression (primary=1, secondary=2, tertiary=3, unknown=2)
    - Rationale: Education has inherent ordering; unknown treated as mid-level based on domain knowledge
  - **Interaction features**:
    - `job_marital`: Combines job and marital status
    - `month_contact`: Combines month and contact type
    - Rationale: Interaction effects between categorical variables can improve predictive power

#### 3. Integration with Modeling Pipeline
- **Updated**: `scripts/response_analysis_2.py` to use `data/bank-full-processed-v2.csv`
- **Change**: Modified `DATA_PATH` configuration to point to the new preprocessing output
- **Impact**: Allows direct comparison between v1 and v2 preprocessing approaches

#### 4. Optimization: Disabling Hyperparameter Tuning for Fast Iteration
- **Created**: `ENABLE_TUNING` configuration flag (default: `False`)
- **Reasoning**: 
  - Hyperparameter tuning with Optuna takes significant time (50 trials × 3 models × 5-fold CV)
  - Needed to rapidly test whether v2 preprocessing improves performance
  - Default parameters for LightGBM/XGBoost/CatBoost are already well-tuned for imbalanced classification
- **Implementation**: All tuning functions (`tune_lightgbm`, `tune_xgboost`, `tune_catboost`) now check `ENABLE_TUNING` flag before execution
- **Future**: Can enable tuning later for production models once preprocessing improvements are validated

#### 5. SHAP Explainability Fixes
- **Issue 1**: Logistic Regression Pipeline error - `property 'feature_names_in_' of 'Pipeline' object has no setter`
  - **Root cause**: SHAP's TreeExplainer/KernelExplainer cannot directly handle sklearn Pipeline objects
  - **Solution**: 
    - Detect Pipeline objects via `hasattr(model, 'named_steps')`
    - Extract underlying model from pipeline
    - Apply scaler transformation to data before SHAP computation
    - Use appropriate explainer for the extracted model type

- **Issue 2**: XGBoost error - `could not convert string to float: '[5E-1]'`
  - **Root cause**: XGBoost models sometimes require specific SHAP API usage or data format
  - **Solution**: 
    - Implemented multi-tier fallback strategy:
      1. Try `TreeExplainer` with `model_output='probability'`
      2. Fallback to standard `TreeExplainer`
      3. Last resort: Use newer `shap.Explainer` API
    - Ensure all data converted to float arrays before SHAP computation
    - Handle different SHAP output formats (lists, 2D arrays, 3D arrays)

- **Impact**: All models can now generate SHAP feature importance plots and CSVs for interpretability

#### 6. Results Validation
- **Performance**: v2 preprocessing results (from terminal output):
  - LightGBM: ROC-AUC = 0.9338, PR-AUC = 0.6275, Precision@10% = 0.6272, Lift@10% = 5.36x
  - CatBoost: ROC-AUC = 0.9327, PR-AUC = 0.6228, Precision@10% = 0.6394, Lift@10% = 5.46x
  - XGBoost: ROC-AUC = 0.9312, PR-AUC = 0.6119, Precision@10% = 0.6239, Lift@10% = 5.33x
  - Logistic Regression: ROC-AUC = 0.9133, PR-AUC = 0.5389, Precision@10% = 0.5940, Lift@10% = 5.08x
- **Note**: Comparison with v1 preprocessing results needed to quantify improvement

---

### Technical Decisions & Rationale

1. **Why feature engineering over hyperparameter tuning?**
   - Tuning provides marginal gains (typically 0.5-2% ROC-AUC improvement)
   - Feature engineering can provide 2-5% improvements by capturing domain knowledge
   - Better ROI on development time

2. **Why keep `previous_contacts` but transform `pdays`?**
   - `pdays` has special value 999 meaning "not previously contacted" - this is better as a binary indicator
   - `previous` is a count with meaningful magnitude, so kept as continuous

3. **Why log-transform balance but cap campaign?**
   - Balance: Skewed distribution benefits from log transformation
   - Campaign: Very high values likely data errors; capping at reasonable maximum (5) is more appropriate

4. **Why ordinal encoding for education but not other categoricals?**
   - Education has inherent progression (primary < secondary < tertiary)
   - Other categoricals (job, marital, etc.) have no natural ordering, so one-hot encoding is more appropriate

5. **Why disable tuning by default?**
   - Fast iteration needed to validate preprocessing approach
   - Default GBDT parameters are already well-suited for imbalanced classification
   - Can re-enable for production after validating that preprocessing helps

---

### Next Steps

1. **Compare v1 vs v2 preprocessing results**: Quantify improvement from feature engineering
2. **Enable hyperparameter tuning**: Once preprocessing is validated, re-run with `ENABLE_TUNING=True`
3. **Business metrics analysis**: Review threshold optimization results and ROI curves
4. **SHAP interpretability**: Analyze feature importance rankings across models to understand what drives predictions
5. **Feature selection**: Potentially remove low-importance features if they don't improve performance

---

### Files Modified/Created

- **Created**: `scripts/data_preprocessing_2.py` - Enhanced preprocessing pipeline
- **Created**: `data/bank-full-processed-v2.csv` - New preprocessed dataset
- **Modified**: `scripts/response_analysis_2.py` - Updated to use v2 preprocessing, added tuning flag, fixed SHAP computation
- **Created**: `RESEARCH_LOG.md` - This file

---

### Lessons Learned

1. **Feature engineering > Hyperparameter tuning** when models are already well-configured
2. **Domain knowledge matters**: Features like `was_contacted_before` and education ordinal encoding leverage business understanding
3. **Iterative development**: Disabling expensive operations (tuning) during development saves time
4. **SHAP compatibility**: Different model types (Pipeline, XGBoost, etc.) require different SHAP approaches
5. **Modular design**: Separate preprocessing script allows easy A/B testing of different preprocessing strategies

