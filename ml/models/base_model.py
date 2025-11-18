"""
BaseModel — Unified abstract class for all ML models
Course-aligned + production safe
"""

from abc import ABC, abstractmethod
import joblib
import os
import json
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all ML models"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.version = self.config.get("version", "1.0.0")
        self.metadata: Dict[str, Any] = {}

    # ============================================================
    # REQUIRED METHODS
    # ============================================================
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """Predict labels / outputs"""
        pass

    def predict_proba(self, X: pd.DataFrame) -> Any:
        """
        Optional: Predict probabilities (for classifiers).

        Default implementation raises if not overridden.
        Clustering models etc. are NOT required to implement this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict_proba()."
        )

    # ============================================================
    # SAVE / LOAD
    # ============================================================
    def save(self, filepath: str) -> None:
        """
        Save underlying model + metadata to disk.

        Accepts both str and Path-like objects.
        """
        # Normalize to string to avoid Path.replace() confusion
        filepath = str(filepath)

        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save the core ML model
        joblib.dump(self.model, filepath, protocol=4)

        # Save metadata next to it
        meta_path = filepath.replace(".pkl", "_meta.json")
        self.metadata = {
            "version": self.version,
            "config": self.config,
            "saved_at": datetime.now().isoformat(),
            "class_name": self.__class__.__name__,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4)

        print(f"✔ Saved model → {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load underlying model + metadata from disk.

        Accepts both str and Path-like objects.
        """
        filepath = str(filepath)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")

        self.model = joblib.load(filepath)
        self.is_trained = True

        meta_path = filepath.replace(".pkl", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                self.metadata = json.load(f)

        print(f"✔ Loaded model → {filepath}")
