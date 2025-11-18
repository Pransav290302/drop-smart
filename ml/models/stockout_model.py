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
        self.feature_names = []

    # ============================================================
    # TRAIN
    # ============================================================
    def train(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)

        self.model.fit(X, y)
        self.is_trained = True

        preds = self.model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, preds)

        print(f"✔ Stockout Risk model trained (AUC = {auc:.4f})")

        self.explainer = shap.TreeExplainer(self.model)

    # ============================================================
    # PREDICT LABEL
    # ============================================================
    def predict(self, X: pd.DataFrame):
        self._check_ready()
        self._check_features(X)
        return self.model.predict(X)

    # ============================================================
    # PREDICT PROBA
    # ============================================================
    def predict_proba(self, X: pd.DataFrame):
        self._check_ready()
        self._check_features(X)
        return self.model.predict_proba(X)[:, 1]

    # ============================================================
    # REQUIRED BY PIPELINE SERVICE
    # ============================================================
    def predict_batch(self, X: pd.DataFrame):
        self._check_ready()
        self._check_features(X)
        return self.model.predict_proba(X)[:, 1]

    # ============================================================
    # SHAP
    # ============================================================
    def explain(self, X: pd.DataFrame):
        self._check_ready()
        self._check_features(X)

        shap_values = self.explainer.shap_values(X)[1]
        base = self.explainer.expected_value

        if isinstance(base, list):
            base = base[1]

        feature_importance = np.abs(shap_values).mean(axis=0)

        per_sample = []
        for i in range(len(X)):
            per_sample.append({
                "risk_score": float(self.predict_proba(X.iloc[[i]])[0]),
                "base_value": float(base),
                "feature_contributions": {
                    f: float(shap_values[i][j])
                    for j, f in enumerate(self.feature_names)
                }
            })

        return {
            "base_value": float(base),
            "feature_importance": {
                self.feature_names[i]: float(feature_importance[i])
                for i in range(len(self.feature_names))
            },
            "per_sample_explanations": per_sample,
            "shap_values": shap_values.tolist()
        }

    # ============================================================
    # INTERNAL CHECKS
    # ============================================================
    def _check_ready(self):
        if not self.is_trained:
            raise RuntimeError("StockoutRiskModel must be trained first.")

    def _check_features(self, X: pd.DataFrame):
        if list(X.columns) != self.feature_names:
            raise ValueError(
                f"Feature mismatch.\nExpected: {self.feature_names}\nGot: {list(X.columns)}"
            )
