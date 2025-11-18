"""
Model Registry — Central place to load ALL trained ML models
Compatible with Training Pipeline V2
"""

from pathlib import Path

from ml.models.viability_model import ViabilityModel
from ml.models.price_model import ConversionModel       # FIXED (correct file)
from ml.models.stockout_model import StockoutRiskModel  # FIXED (correct file)
from ml.models.clustering_model import ClusteringModel


class ModelRegistry:
    """
    Loads and stores all ML models:
    - ViabilityModel
    - ConversionModel
    - StockoutRiskModel
    - ClusteringModel
    """

    def __init__(self, root: str = "data/models"):
        self.root = Path(root)

        # instantiate empty objects
        self.viability = ViabilityModel()
        self.conversion = ConversionModel()
        self.stockout = StockoutRiskModel()
        self.clustering = ClusteringModel()

    def load_all(self):
        """Load ALL trained models from disk."""

        # -----------------------------------------
        # 1. Viability Model
        # -----------------------------------------
        viability_path = self.root / "viability" / "model.pkl"
        self.viability.load(viability_path)

        # -----------------------------------------
        # 2. Conversion Model
        # -----------------------------------------
        conv_path = self.root / "price_optimizer" / "conversion_model.pkl"
        self.conversion.load(conv_path)

        # -----------------------------------------
        # 3. Stockout Risk Model
        # -----------------------------------------
        stockout_path = self.root / "stockout_risk" / "model.pkl"
        self.stockout.load(stockout_path)

        # -----------------------------------------
        # 4. Product Clustering
        # -----------------------------------------
        cluster_path = self.root / "clustering" / "kmeans.pkl"
        self.clustering.load(cluster_path)

        print("🔥 All ML models loaded successfully!")

    def get_all(self):
        """Return dictionary of loaded models."""
        return {
            "viability": self.viability,
            "conversion": self.conversion,
            "stockout": self.stockout,
            "clustering": self.clustering,
        }
