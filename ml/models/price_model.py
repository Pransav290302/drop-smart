"""
Conversion Probability Model — Predicts probability of purchase at a given price.
Aligned with course topics: LogisticRegression + RandomForest + SHAP.
"""

import numpy as np
import pandas as pd
import shap
from typing import Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from ml.models.base_model import BaseModel


class ConversionModel(BaseModel):
    """
    Predicts conversion probability p(buy | price, features)
    Fully compatible with:
    - BaseModel (train, predict, predict_proba)
    - PriceOptimizer (predict_for_price)
    """

    def __init__(self, model_type="logistic_regression", config=None):
        super().__init__(config)

        self.model_type = model_type.lower()

        # REQUIRED BY PRICE OPTIMIZER
        self.price_feature_name = "price"

        self.scaler = None
        self.feature_names: List[str] = []
        self.explainer = None

        # ----------------------------------------------------------
        # MODEL SELECTION
        # ----------------------------------------------------------
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=7,
                class_weight="balanced",
                random_state=42
            )
        else:
            self.scaler = StandardScaler()
            self.model = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )

    # ============================================================
    # TRAIN
    # ============================================================
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:

        if self.price_feature_name not in X.columns:
            raise ValueError(f"Missing price column '{self.price_feature_name}'")

        self.feature_names = list(X.columns)

        # Logistic Regression + calibration
        if self.model_type == "logistic_regression":
            X_scaled = self.scaler.fit_transform(X)
            base = self.model.fit(X_scaled, y)

            self.model = CalibratedClassifierCV(base, cv=3)
            self.model.fit(X_scaled, y)

            preds = self.model.predict_proba(X_scaled)[:, 1]

        else:
            # Random Forest version
            self.model.fit(X, y)
            preds = self.model.predict_proba(X)[:, 1]

        self.is_trained = True
        print("✔ Conversion model trained.")

        # Initialize SHAP
        self._init_shap(X)

    # ============================================================
    # PREDICT LABEL
    # ============================================================
    def predict(self, X: pd.DataFrame) -> Any:
        prob = self.predict_proba(X)
        return (prob > 0.5).astype(int)

    # ============================================================
    # PREDICT PROBABILITY
    # ============================================================
    def predict_proba(self, X: pd.DataFrame) -> Any:
        self._check_ready()
        self._check_features(X)

        if self.model_type == "logistic_regression":
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)[:, 1]

        return self.model.predict_proba(X)[:, 1]

    # ============================================================
    # PRICE SIMULATION  — USED BY PriceOptimizer
    # ============================================================
    def predict_for_price(self, price: float, features: pd.DataFrame) -> float:

        X = features.copy()

        # Set the price value
        X[self.price_feature_name] = float(price)

        # Make sure all columns exist
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0

        # Correct order
        X = X[self.feature_names]

        return float(self.predict_proba(X)[0])

    # ============================================================
    # SHAP
    # ============================================================
    def _init_shap(self, X: pd.DataFrame):
        try:
            sample = X.sample(min(100, len(X)), random_state=42)

            if self.model_type == "logistic_regression":
                sample_scaled = self.scaler.transform(sample)

                # Fixed key: estimator not base_estimator
                base_est = self.model.calibrated_classifiers_[0].estimator
                self.explainer = shap.LinearExplainer(base_est, sample_scaled)

            else:
                self.explainer = shap.TreeExplainer(self.model)

        except Exception as e:
            print("⚠ SHAP initialization failed:", e)
            self.explainer = None

    # ============================================================
    # INTERNAL CHECKS
    # ============================================================
    def _check_ready(self):
        if not self.is_trained:
            raise RuntimeError("ConversionModel must be trained first.")

    def _check_features(self, X: pd.DataFrame):
        if list(X.columns) != self.feature_names:
            raise ValueError(
                "Feature mismatch.\n"
                f"Expected: {self.feature_names}\n"
                f"Got:      {list(X.columns)}"
            )
