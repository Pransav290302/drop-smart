"""
Viability Pipeline — CLEAN VERSION (B2-A)
Fully aligned with new ViabilityModel (RandomForest + SHAP)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from ml.models.viability_model import ViabilityModel

logger = logging.getLogger(__name__)


class ViabilityPipeline:
    """
    Clean viability pipeline for:
    - Training
    - Evaluation
    - Batch predictions
    - SHAP explanations
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize pipeline with NEW course-aligned ViabilityModel."""
        self.model = ViabilityModel(config=config)
        self.evaluation_metrics: Dict[str, Any] = {}

    # =========================================================
    # DATA PREPARATION
    # =========================================================
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = "sold_within_30_days",
        test_size: float = 0.25,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

        if target_column not in df.columns:
            raise ValueError(f"Missing target column: {target_column}")

        X = df.drop(columns=[target_column])
        y = df[target_column].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, y_train, X_test, y_test

    # =========================================================
    # TRAINING
    # =========================================================
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        logger.info("Training viability model...")
        self.model.train(X_train, y_train)
        logger.info("Training complete.")

    # =========================================================
    # EVALUATION
    # =========================================================
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        metrics = {}

        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_test, y_proba))
        metrics["brier_score"] = float(brier_score_loss(y_test, y_proba))
        metrics["classification_report"] = classification_report(
            y_test, y_pred, output_dict=True
        )

        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }

        # ROC Curve
        fpr, tpr, th = roc_curve(y_test, y_proba)
        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": th.tolist(),
        }

        # PR Curve
        precision, recall, th2 = precision_recall_curve(y_test, y_proba)
        metrics["pr_curve"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": th2.tolist(),
        }

        # Calibration Curve
        frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
        metrics["calibration_curve"] = {
            "fraction_of_positives": frac_pos.tolist(),
            "mean_predicted_value": mean_pred.tolist(),
        }

        self.evaluation_metrics = metrics
        logger.info(f"Evaluation ROC-AUC: {metrics['roc_auc']:.4f}")

        return metrics

    # =========================================================
    # PREDICTION + SHAP EXPLANATION
    # =========================================================
    def predict(
        self,
        X: pd.DataFrame,
        include_explanations: bool = True
    ) -> Dict[str, Any]:

        proba = self.model.predict_proba(X)
        pred = (proba >= 0.5).astype(int)

        output = {
            "predictions": pred.tolist(),
            "probabilities": proba.tolist(),
        }

        if include_explanations:
            try:
                explanations = self.model.explain(X)
                output["explanations"] = explanations
            except Exception as e:
                logger.warning(f"SHAP failed: {e}")
                output["explanations"] = None

        return output

    # =========================================================
    # BATCH FORMATTER
    # =========================================================
    def predict_batch(
        self,
        items: List[Dict[str, Any]],
        feature_columns: List[str]
    ) -> List[Dict[str, Any]]:

        df = pd.DataFrame(items)
        X = df[feature_columns]

        results = self.predict(X, include_explanations=False)

        formatted = []
        for i, item in enumerate(items):
            formatted.append({
                "sku": item.get("sku", f"product_{i}"),
                "viability_score": float(results["probabilities"][i]),
                "prediction": int(results["predictions"][i]),
            })

        return formatted

    # =========================================================
    # SAVE / LOAD
    # =========================================================
    def save_model(self, filepath: Union[str, Path]):
        self.model.save(filepath)

    def load_model(self, filepath: Union[str, Path]):
        self.model.load(filepath)

