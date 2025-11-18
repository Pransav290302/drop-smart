"""ML Pipeline service for orchestrating model calls"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from backend.app.core.config import settings

# ML model imports
from ml.models.viability_model import ViabilityModel
from ml.models.price_model import ConversionModel
from ml.models.stockout_model import StockoutRiskModel

# NEW TF-IDF + KMeans clustering model (correct import)
from ml.models.clustering_model import ClusteringModel

from ml.services.price_optimizer import PriceOptimizer
from ml.pipelines.viability_pipeline import ViabilityPipeline  # noqa

# Data processing imports
from ml.data.normalization import DataNormalizer
from ml.features.engineering import engineer_features
from ml.config import get_schema_config

logger = logging.getLogger(__name__)


class MLPipelineService:
    """
    Service orchestrating:
    - viability prediction
    - price optimization
    - stockout risk
    - TF-IDF clustering
    """

    def __init__(self):
        self.viability_model: Optional[ViabilityModel] = None
        self.conversion_model: Optional[ConversionModel] = None
        self.price_optimizer: Optional[PriceOptimizer] = None
        self.stockout_model: Optional[StockoutRiskModel] = None
        self.clusterer: Optional[ClusteringModel] = None

        # Model paths
        self.viability_model_path = settings.models_dir / "viability" / "model.pkl"
        self.conversion_model_path = settings.models_dir / "price_optimizer" / "conversion_model.pkl"
        self.stockout_model_path = settings.models_dir / "stockout_risk" / "model.pkl"
        self.clustering_model_path = settings.models_dir / "clustering" / "model.pkl"

        # Normalizer
        try:
            self.normalizer = DataNormalizer(config=get_schema_config())
        except Exception:
            self.normalizer = DataNormalizer()

        self._load_models()

    # ------------------------------------------------------------------
    # FIXED: text builder (fully replaces prepare_texts_for_clustering)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_product_text(product: Dict[str, Any]) -> str:
        parts: List[str] = []

        for key in ("product_name", "name", "title"):
            v = product.get(key)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())

        for key in ("description", "category"):
            v = product.get(key)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())

        if not parts:
            parts.append(str(product.get("sku", "unknown")))

        return " ".join(parts).lower()

    # ------------------------------------------------------------------
    # LOAD MODELS
    # ------------------------------------------------------------------

    def _load_models(self) -> None:

        # Viability
        try:
            if self.viability_model_path.exists():
                self.viability_model = ViabilityModel()
                self.viability_model.load(self.viability_model_path)
                logger.info("Viability model loaded")
        except Exception as e:
            logger.error(f"Failed to load viability model: {e}")

        # Conversion + price optimizer
        try:
            if self.conversion_model_path.exists():
                self.conversion_model = ConversionModel()
                self.conversion_model.load(self.conversion_model_path)
                self.price_optimizer = PriceOptimizer(self.conversion_model)
                logger.info("Conversion + PriceOptimizer loaded")
        except Exception as e:
            logger.error(f"Failed to load conversion model: {e}")

        # Stockout
        try:
            if self.stockout_model_path.exists():
                self.stockout_model = StockoutRiskModel()
                self.stockout_model.load(self.stockout_model_path)
                logger.info("Stockout model loaded")
        except Exception as e:
            logger.error(f"Failed to load stockout model: {e}")

        # Clustering TF-IDF
        try:
            if self.clustering_model_path.exists():
                self.clusterer = ClusteringModel()
                self.clusterer.load(self.clustering_model_path)
                logger.info("TF-IDF clustering model loaded")
        except Exception as e:
            logger.error(f"Failed to load clustering model: {e}")

    # ------------------------------------------------------------------
    # FEATURE PREPARATION
    # ------------------------------------------------------------------

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features_df = df.copy()

        # Normalize
        try:
            features_df = self.normalizer.normalize_all(features_df)
        except Exception:
            pass

        # feature engineering
        try:
            from ml.config import get_model_config
            cfg = get_model_config()
            features_df = engineer_features(features_df, config=cfg.get("features", {}))
        except Exception:
            # basic fallback
            features_df["landed_cost"] = (
                features_df.get("cost", 0)
                + features_df.get("shipping_cost", 0)
                + features_df.get("duties", 0)
            )
            features_df["margin_percent"] = (
                (features_df.get("price", 0) - features_df["landed_cost"])
                / features_df.get("price", 1)
                * 100
            ).fillna(0)

        # Availability encoding
        if "availability" in features_df.columns:
            availability_map = {
                "in_stock": 1.0,
                "low_stock": 0.5,
                "out_of_stock": 0.0,
                "pre_order": 0.3,
            }
            features_df["availability_encoded"] = (
                features_df["availability"].map(availability_map).fillna(0.0)
            )

        return features_df

    # ------------------------------------------------------------------
    # VIABILITY
    # ------------------------------------------------------------------

    def predict_viability(self, products, top_k=None):
        if self.viability_model is None:
            raise ValueError("Viability model missing")

        df = pd.DataFrame(products)
        fdf = self.prepare_features(df)

        # fill missing
        for col in self.viability_model.feature_names:
            if col not in fdf.columns:
                fdf[col] = 0.0

        X = fdf[self.viability_model.feature_names]
        scores = self.viability_model.predict_viability_score(X)

        results = []
        for i, p in enumerate(products):
            s = float(scores[i])
            cls = "high" if s >= 0.7 else "medium" if s >= 0.4 else "low"
            results.append({
                "sku": p.get("sku", f"product_{i}"),
                "viability_score": s,
                "viability_class": cls,
            })

        results.sort(key=lambda r: r["viability_score"], reverse=True)
        return results[:top_k] if top_k else results

    # ------------------------------------------------------------------
    # PRICE OPTIMIZATION
    # ------------------------------------------------------------------

    def optimize_price(self, products, min_margin_percent=0.15, enforce_map=True):
        if self.price_optimizer is None:
            raise ValueError("Conversion model missing")

        self.price_optimizer.min_margin_percent = min_margin_percent
        self.price_optimizer.enforce_map = enforce_map

        return self.price_optimizer.optimize_batch(products)

    # ------------------------------------------------------------------
    # STOCKOUT RISK
    # ------------------------------------------------------------------

    def predict_stockout_risk(self, products):

        if self.stockout_model is None:
            raise ValueError("Stockout model missing")

        df = pd.DataFrame(products)
        fdf = self.prepare_features(df)

        for col in self.stockout_model.feature_names:
            if col not in fdf.columns:
                fdf[col] = 0.0

        X = fdf[self.stockout_model.feature_names]
        scores = self.stockout_model.predict_batch(X)

        results = []
        for i, p in enumerate(products):
            s = float(scores[i])
            lvl = "high" if s >= 0.7 else "medium" if s >= 0.4 else "low"
            results.append({
                "sku": p.get("sku", f"product_{i}"),
                "risk_score": s,
                "risk_level": lvl,
            })

        return results

    # ------------------------------------------------------------------
    # CLUSTERING (TF-IDF + KMeans)
    # ------------------------------------------------------------------

    def get_cluster_assignments(self, products):

        if self.clusterer is None:
            return [None] * len(products)

        texts = [self._build_product_text(p) for p in products]

        try:
            labels = self.clusterer.predict(texts)
            return labels.tolist()
        except Exception:
            return [None] * len(products)

    # ------------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------------

    def process_complete_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:

        products = df.to_dict("records")
        out = df.copy()

        # viability
        try:
            v = self.predict_viability(products)
            vd = {x["sku"]: x for x in v}
            out["viability_score"] = out["sku"].map(lambda s: vd.get(s, {}).get("viability_score", 0.0))
            out["viability_class"] = out["sku"].map(lambda s: vd.get(s, {}).get("viability_class", "low"))
        except Exception:
            out["viability_score"] = 0
            out["viability_class"] = "low"

        # pricing
        try:
            p = self.optimize_price(products)
            pdict = {x["sku"]: x for x in p}
            out["recommended_price"] = out["sku"].map(lambda s: pdict.get(s, {}).get("recommended_price", out["price"]))
            out["expected_profit"] = out["sku"].map(lambda s: pdict.get(s, {}).get("expected_profit", 0))
        except Exception:
            out["recommended_price"] = out["price"]
            out["expected_profit"] = 0

        # stockout
        try:
            r = self.predict_stockout_risk(products)
            rd = {x["sku"]: x for x in r}
            out["stockout_risk_score"] = out["sku"].map(lambda s: rd.get(s, {}).get("risk_score", 0))
            out["stockout_risk_level"] = out["sku"].map(lambda s: rd.get(s, {}).get("risk_level", "low"))
        except Exception:
            out["stockout_risk_score"] = 0
            out["stockout_risk_level"] = "low"

        # clustering
        out["cluster_id"] = self.get_cluster_assignments(products)

        # margin
        out["landed_cost"] = out["cost"] + out["shipping_cost"] + out["duties"]
        out["margin_percent"] = ((out["recommended_price"] - out["landed_cost"]) / out["recommended_price"] * 100).fillna(0)

        out = out.sort_values("viability_score", ascending=False)
        out["rank"] = range(1, len(out) + 1)

        return out


pipeline_service = MLPipelineService()
