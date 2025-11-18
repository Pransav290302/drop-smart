"""
Stockout Risk Model — Predicts probability of high stockout risk
Course-Aligned (RandomForest + SHAP)
"""

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from ml.models.base_model import BaseModel


class StockoutRiskModel(BaseModel):
    """
    Predicts probability of HIGH stockout risk.
    Fully compatible with BaseModel (implements predict()).
    """

    def __init__(self, config=None):
        super().__init__(config)

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight="balanced",
            random_state=42
        )

        self.explainer = None
        self.feature_names: list[str] = []
        self.is_trained = False

    # ============================================================
    # TRAIN
    # ============================================================
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train RF classifier and SHAP explainer."""
        self.feature_names = list(X.columns)

        self.model.fit(X, y)
        self.is_trained = True

        preds = self.model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, preds)

        print(f"✔ Stockout Risk model trained (AUC = {auc:.4f})")

        # SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

    # ============================================================
    # REQUIRED BY BaseModel
    # ============================================================
    def predict(self, X: pd.DataFrame):
        """
        Required abstract method → returns hard labels (0/1).
        """
        self._check_ready()
        X_checked = self._align(X)
        return self.model.predict(X_checked)

    # ============================================================
    # PROBABILITY PREDICTION
    # ============================================================
    def predict_proba(self, X: pd.DataFrame):
        """Return probability of high stockout risk."""
        self._check_ready()
        X_checked = self._align(X)
        return self.model.predict_proba(X_checked)[:, 1]

    # For pipeline service compatibility
    def predict_batch(self, X: pd.DataFrame):
        """Alias for pipeline service."""
        return self.predict_proba(X)

    # ============================================================
    # SHAP EXPLANATION
    # ============================================================
    def explain(self, X: pd.DataFrame):
        self._check_ready()
        X_checked = self._align(X)

        shap_values = self.explainer.shap_values(X_checked)[1]
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]

        feature_importance = np.abs(shap_values).mean(axis=0)

        per_sample = []
        for i in range(len(X_checked)):
            per_sample.append({
                "risk_score": float(self.predict_proba(X_checked.iloc[[i]])[0]),
                "base_value": float(base_value),
                "feature_contributions": {
                    f: float(shap_values[i][j])
                    for j, f in enumerate(self.feature_names)
                }
            })

        return {
            "base_value": float(base_value),
            "feature_importance": {
                self.feature_names[i]: float(feature_importance[i])
                for i in range(len(self.feature_names))
            },
            "per_sample_explanations": per_sample,
            "shap_values": shap_values.tolist()
        }

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================
    def _check_ready(self):
        if not self.is_trained:
            raise RuntimeError("StockoutRiskModel must be trained first.")

    def _align(self, X: pd.DataFrame):
        """
        Ensure X columns match training features exactly.
        Fill missing columns with 0.0.
        Ignore extra columns.
        Prevents warnings:
        'X does not have valid feature names'
        """
        X = X.copy()

        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0

        X = X[self.feature_names]

        return X
