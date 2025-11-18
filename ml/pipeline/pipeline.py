"""
Complete ML Pipeline for DropSmart
----------------------------------
Runs:

✔ ViabilityModel
✔ ConversionModel
✔ StockoutRiskModel
✔ ClusteringModel
✔ Feature Engineering
✔ Ranking

This is the master pipeline your FastAPI backend will call.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import os

from ml.models.viability_model import ViabilityModel
from ml.models.price_model import ConversionModel
from ml.models.stockout_model import StockoutRiskModel
from ml.models.clustering_model import ClusteringModel

logger = logging.getLogger(__name__)


class MLPipeline:
    """End-to-end ML pipeline used by the DropSmart backend."""

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)

        # Load models
        self.viability_model = ViabilityModel()
        self.conversion_model = ConversionModel()
        self.stockout_model = StockoutRiskModel()
        self.clustering_model = ClusteringModel()

        self._load_all_models()

    # ----------------------------------------------------------------------
    # LOAD MODELS
    # ----------------------------------------------------------------------
    def _load_all_models(self):
        logger.info("Loading trained models...")

        self.viability_model.load(self.model_dir / "viability/model.pkl")
        self.conversion_model.load(self.model_dir / "price_optimizer/conversion_model.pkl")
        self.stockout_model.load(self.model_dir / "stockout_risk/model.pkl")
        self.clustering_model.load(self.model_dir / "clustering/kmeans.pkl")

        logger.info("✓ All models loaded successfully")

    # ----------------------------------------------------------------------
    # FEATURE ENGINEERING
    # ----------------------------------------------------------------------
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["landed_cost"] = df["cost"] + df["shipping_cost"] + df["duties"]
        df["margin"] = (df["price"] - df["landed_cost"]) / df["price"]
        df["margin"] = df["margin"].clip(lower=0)

        return df

    # ----------------------------------------------------------------------
    # RUN PIPELINE
    # ----------------------------------------------------------------------
    def run(self, input_file: Path) -> Dict[str, Any]:
        logger.info(f"Running ML Pipeline on: {input_file}")

        # 1️⃣ Load Excel
        df = pd.read_excel(input_file)
        df = self._feature_engineering(df)

        FEATURES = [
            "price", "cost", "shipping_cost", "duties",
            "lead_time_days", "stock", "inventory", "quantity",
            "demand", "past_sales",
            "weight_kg", "length_cm", "width_cm", "height_cm",
            "margin", "supplier_reliability_score"
        ]
        X = df[FEATURES].fillna(0)

        # 2️⃣ Run predictions
        viability_scores = self.viability_model.predict_proba(X)
        conversion_scores = self.conversion_model.predict_proba(X)
        stockout_scores = self.stockout_model.predict_risk_score(X)

        # 3️⃣ Clustering
        product_texts = (
            df["product_name"].astype(str) + " " + df["description"].astype(str)
        ).tolist()

        cluster_ids = self.clustering_model.predict(product_texts)

        # 4️⃣ Ranking
        rank_scores = (
            viability_scores * 0.55 +          # most important
            conversion_scores * 0.30 +
            (1 - stockout_scores) * 0.15       # lower risk = higher score
        )

        ranking = np.argsort(-rank_scores)  # descending

        # 5️⃣ Build final result list
        results = []
        for i in range(len(df)):
            results.append({
                "sku": df.iloc[i]["sku"],
                "product_name": df.iloc[i]["product_name"],
                "viability_score": float(viability_scores[i]),
                "conversion_probability": float(conversion_scores[i]),
                "stockout_risk": float(stockout_scores[i]),
                "cluster_id": int(cluster_ids[i]),
                "rank": int(np.where(ranking == i)[0][0] + 1)
            })

        logger.info("Pipeline completed successfully.")

        return {
            "status": "success",
            "total_products": len(df),
            "results": results
        }
