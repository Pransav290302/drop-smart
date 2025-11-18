# Viability Model Implementation

## ✅ Complete Implementation

### Files Created

1. **`ml/models/viability_model.py`** - Core viability model class
2. **`ml/pipelines/viability_pipeline.py`** - Training and inference pipeline

## ViabilityModel Class

### Features

- **Dual Model Support**: LightGBM (primary) and Logistic Regression (baseline)
- **SHAP Explainability**: Full SHAP integration for per-product explanations
- **Probability Prediction**: Predicts P(sale within 30 days) as per FR-5
- **Model Persistence**: Save/load functionality
- **Feature Importance**: Built-in feature importance extraction

### Key Methods

#### `fit(X, y, validation_data=None)`
- Trains the model on provided data
- Supports early stopping with validation data (LightGBM)
- Automatically initializes SHAP explainer after training
- Handles both LightGBM and Logistic Regression training

#### `predict_proba(X)`
- Returns probability predictions [P(no_sale), P(sale)]
- Convenience method `predict_viability_score()` returns only P(sale)

#### `explain(X, return_shap_values=True)`
- Generates SHAP explanations for predictions
- Returns:
  - Raw SHAP values
  - Base value (expected model output)
  - Feature importance (mean absolute SHAP)
  - Per-sample explanations with feature contributions

#### `save(filepath)` / `load(filepath)`
- Saves/loads complete model state including scaler and feature names
- Uses pickle for serialization

### Model Configuration

Supports configuration via dictionary or config file:

```python
config = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_samples": 20,
    "class_weight": "balanced",
    "random_state": 42,
}
```

### SHAP Integration

- **LightGBM**: Uses `TreeExplainer` (fast, exact)
- **Logistic Regression**: Uses `LinearExplainer`
- Background sampling for faster computation
- Per-sample feature contribution explanations

## ViabilityPipeline Class

### Features

- **Data Preparation**: Train/test splitting with stratification
- **Model Training**: Wraps model training with validation support
- **Comprehensive Evaluation**: Full evaluation metrics
- **Batch Prediction**: Predict for multiple products with explanations
- **Evaluation Metrics**: ROC-AUC, PR-AUC, Brier score, calibration curves

### Key Methods

#### `prepare_data(df, target_column, test_size=0.2)`
- Splits data into train/test sets
- Handles binary target validation
- Returns (X_train, y_train, X_test, y_test)

#### `train(X_train, y_train, X_val=None, y_val=None)`
- Trains the model
- Supports validation data for early stopping

#### `evaluate(X_test, y_test, return_predictions=False)`
- Comprehensive evaluation with metrics:
  - ROC-AUC score
  - Precision-Recall AUC
  - Brier score (calibration)
  - Classification report
  - Confusion matrix
  - ROC and PR curves
  - Calibration curve data

#### `predict(X, include_explanations=True)`
- Makes predictions with optional SHAP explanations
- Returns predictions, probabilities, and explanations

#### `predict_batch(products, feature_columns, include_explanations=True)`
- Predicts viability for batch of products
- Returns formatted results per product with SHAP values

### Evaluation Metrics

The pipeline calculates:
- **ROC-AUC**: Area under ROC curve (target: ≥ 0.80 per PRD)
- **PR-AUC**: Precision-Recall AUC (important for imbalanced data)
- **Brier Score**: Calibration metric (lower is better)
- **Classification Metrics**: Precision, recall, F1-score
- **Calibration Curve**: Model calibration visualization data

## Usage Examples

### Training a Model

```python
from ml.pipelines.viability_pipeline import ViabilityPipeline
import pandas as pd

# Initialize pipeline
pipeline = ViabilityPipeline(model_type="lightgbm")

# Prepare data (assumes df has features + "sold_within_30_days" column)
X_train, y_train, X_test, y_test = pipeline.prepare_data(df)

# Train model
pipeline.train(X_train, y_train)

# Evaluate
metrics = pipeline.evaluate(X_test, y_test)
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"PR-AUC: {metrics['pr_auc']:.4f}")

# Save model
pipeline.save_model("data/models/viability/model.pkl")
```

### Making Predictions

```python
# Load model
pipeline.load_model("data/models/viability/model.pkl")

# Predict for new products
X_new = pd.DataFrame({
    "cost": [25.0, 30.0],
    "price": [49.99, 59.99],
    "margin_percent": [30.0, 35.0],
    # ... other features
})

results = pipeline.predict(X_new, include_explanations=True)

# Access predictions
for i, prob in enumerate(results["probabilities"]):
    print(f"Product {i}: Viability score = {prob:.4f}")
    print(f"SHAP values: {results['explanations']['per_sample_explanations'][i]}")
```

### Batch Prediction

```python
products = [
    {
        "sku": "SKU001",
        "cost": 25.0,
        "price": 49.99,
        "margin_percent": 30.0,
        # ... other features
    },
    # ... more products
]

feature_columns = ["cost", "price", "margin_percent", ...]  # All feature names

results = pipeline.predict_batch(products, feature_columns, include_explanations=True)

for result in results:
    print(f"{result['sku']}: {result['viability_score']:.4f} ({result['viability_class']})")
    print(f"SHAP: {result['shap_values']}")
```

## Integration with FastAPI

The model can be integrated into the FastAPI backend:

```python
from ml.pipelines.viability_pipeline import ViabilityPipeline

# Initialize pipeline (load trained model)
pipeline = ViabilityPipeline()
pipeline.load_model("data/models/viability/model.pkl")

# In API endpoint
def predict_viability(products):
    feature_columns = pipeline.model.feature_names
    results = pipeline.predict_batch(products, feature_columns, include_explanations=True)
    return results
```

## Requirements Met

✅ **FR-5**: Predicts P(sale within 30 days)  
✅ **FR-6**: Supports LightGBM + Logistic Regression baseline  
✅ **FR-7**: Full SHAP explainability integration  

## Model Performance Targets (from PRD)

- **ROC-AUC**: ≥ 0.80
- **PR-AUC**: Important for imbalanced data
- **Calibration**: Brier score for model calibration

## Next Steps

1. Train model on actual data
2. Tune hyperparameters
3. Evaluate on test set
4. Integrate with FastAPI backend
5. Replace stub functions in `backend/app/api/routes.py`

