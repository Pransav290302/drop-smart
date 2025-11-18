"""
ConversionModel V3 — Binary Logistic Conversion Classifier
Trains on FEATURES from training pipeline V3.

- conversion_flag (0/1)
- LogisticRegression
- Provides:
    * predict()
    * predict_proba()
    * predict_conversion_probability()
    * predict_for_price()
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.models.base_model import BaseModel


class ConversionModel(BaseModel):
    """
    Logistic regression classification model for conversion probability.

    FIXED:
    - Implements .predict() → required by BaseModel
    - predict_for_price has NO wrong arguments
    - proper feature preselection
    """

    # Training pipeline V3 FEATURES
    FEATURES = [
        "price", "cost", "shipping_cost", "duties",
        "lead_time_days", "stock", "inventory", "quantity",
        "demand", "past_sales",
        "weight_kg", "length_cm", "width_cm", "height_cm",
        "margin", "supplier_reliability_score"
    ]

    def __init__(self, config=None):
        super().__init__(config)
        self.model = LogisticRegression(
            max_iter=2000,
            solver="lbfgs"
        )
        self.is_trained = False

    # ---------------------------------------------------
    # TRAIN
    # ---------------------------------------------------
    def train(self, X: pd.DataFrame, y: pd.Series):
        X = X[self.FEATURES].fillna(0)
        self.model.fit(X, y)
        self.is_trained = True

    # ---------------------------------------------------
    # Required by BaseModel → FIX
    # ---------------------------------------------------
    def predict(self, X: pd.DataFrame):
        """Return binary classification (0 or 1)."""
        X = X[self.FEATURES].fillna(0)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        """Return probability of conversion."""
        X = X[self.FEATURES].fillna(0)
        return self.model.predict_proba(X)[:, 1]

    # ---------------------------------------------------
    # API for PriceOptimizer
    # ---------------------------------------------------
    def predict_conversion_probability(self, product: dict, price: float) -> float:
        """
        Compute conversion probability at a given price.
        Called by PriceOptimizer.
        """
        row = {feat: product.get(feat, 0) for feat in self.FEATURES}
        row["price"] = price
        row["margin"] = (
            (price - (product.get("cost", 0)
                      + product.get("shipping_cost", 0)
                      + product.get("duties", 0)))
            / price
        )

        X = pd.DataFrame([row])
        return float(self.predict_proba(X)[0])

    def predict_for_price(self, product: dict, price: float) -> float:
        """Alias used inside PriceOptimizer — clean, no kwargs."""
        return self.predict_conversion_probability(product, price)

    # ---------------------------------------------------
    # SAVE / LOAD
    # ---------------------------------------------------
    def save(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "config": self.config,
                    "is_trained": self.is_trained,
                },
                f,
            )

    def load(self, filepath):
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.config = data.get("config", {})
        self.is_trained = data.get("is_trained", True)
