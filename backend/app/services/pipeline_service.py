"""ML Pipeline service for orchestrating model calls"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from backend.app.core.config import settings

# ML model imports
from ml.models.viability_model import ViabilityModel
from ml.models.price_model import ConversionModel
from ml.models.stockout_model import StockoutRiskModel
from ml.models.clustering.clustering import (
    ProductClustering,
    prepare_texts_for_clustering
)

from ml.services.price_optimizer import PriceOptimizer
from ml.pipelines.viability_pipeline import ViabilityPipeline

# Data processing imports
from ml.data.normalization import DataNormalizer
from ml.features.engineering import engineer_features
from ml.config import get_schema_config

logger = logging.getLogger(__name__)


class MLPipelineService:
    """
    Service for orchestrating ML model calls.
    
    Handles:
    - Loading trained models
    - Processing product data
    - Calling appropriate models
    - Combining results
    """
    
    def __init__(self):
        """Initialize pipeline service with model paths."""
        self.viability_model: Optional[ViabilityModel] = None
        self.conversion_model: Optional[ConversionModel] = None
        self.price_optimizer: Optional[PriceOptimizer] = None
        self.stockout_model: Optional[StockoutRiskModel] = None
        self.clusterer: Optional[ProductClustering] = None
        
        # Model paths
        self.viability_model_path = settings.models_dir / "viability" / "model.pkl"
        self.conversion_model_path = settings.models_dir / "price_optimizer" / "conversion_model.pkl"
        self.stockout_model_path = settings.models_dir / "stockout_risk" / "model.pkl"
        self.clustering_model_path = settings.models_dir / "clustering" / "model.pkl"
        
        # Initialize data normalizer
        try:
            schema_config = get_schema_config()
            self.normalizer = DataNormalizer(config=schema_config)
        except Exception as e:
            logger.warning(f"Failed to initialize data normalizer: {e}. Using defaults.")
            self.normalizer = DataNormalizer()
        
        # Load models if they exist
        self._load_models()
    
    def _load_models(self) -> None:
        """Load trained models if they exist."""
        try:
            # Load viability model
            if self.viability_model_path.exists():
                self.viability_model = ViabilityModel()
                self.viability_model.load(self.viability_model_path)
                logger.info("Viability model loaded")
            else:
                logger.warning(f"Viability model not found at {self.viability_model_path}")
        except Exception as e:
            logger.error(f"Failed to load viability model: {e}")
        
        try:
            # Load conversion model
            if self.conversion_model_path.exists():
                self.conversion_model = ConversionModel()
                self.conversion_model.load(self.conversion_model_path)
                # Initialize price optimizer
                self.price_optimizer = PriceOptimizer(conversion_model=self.conversion_model)
                logger.info("Conversion model and price optimizer loaded")
            else:
                logger.warning(f"Conversion model not found at {self.conversion_model_path}")
        except Exception as e:
            logger.error(f"Failed to load conversion model: {e}")
        
        try:
            # Load stockout risk model
            if self.stockout_model_path.exists():
                self.stockout_model = StockoutRiskModel()
                self.stockout_model.load(self.stockout_model_path)
                logger.info("Stockout risk model loaded")
            else:
                logger.warning(f"Stockout risk model not found at {self.stockout_model_path}")
        except Exception as e:
            logger.error(f"Failed to load stockout risk model: {e}")
        
        try:
            # Load clustering model
            if self.clustering_model_path.exists():
                self.clusterer = ProductClustering()
                self.clusterer.load(self.clustering_model_path)
                logger.info("Clustering model loaded")
            else:
                logger.warning(f"Clustering model not found at {self.clustering_model_path}")
        except Exception as e:
            logger.error(f"Failed to load clustering model: {e}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from DataFrame for ML models.
        
        FR-3: Normalize currencies, dimensions, and weights
        FR-4: Compute derived fields (volumetric weight, seasonality, lead-time buckets)
        
        Args:
            df: Raw product DataFrame
            
        Returns:
            DataFrame with normalized and engineered features
        """
        features_df = df.copy()
        
        # Step 1: Normalize data (FR-3)
        try:
            features_df = self.normalizer.normalize_all(features_df)
            logger.debug("Data normalization complete")
        except Exception as e:
            logger.warning(f"Data normalization failed: {e}. Continuing without normalization.")
        
        # Step 2: Engineer features (FR-4)
        try:
            # Get feature engineering config
            from ml.config import get_model_config
            model_config = get_model_config()
            feature_config = model_config.get("features", {})
            
            features_df = engineer_features(features_df, config=feature_config)
            logger.debug("Feature engineering complete")
        except Exception as e:
            logger.warning(f"Feature engineering failed: {e}. Using basic features only.")
            # Fallback to basic features
            features_df["landed_cost"] = (
                features_df.get("cost", 0) +
                features_df.get("shipping_cost", 0) +
                features_df.get("duties", 0)
            )
            features_df["margin_percent"] = (
                ((features_df.get("price", 0) - features_df["landed_cost"]) / features_df.get("price", 1)) * 100
            ).fillna(0)
        
        # Encode availability
        availability_map = {
            "in_stock": 1,
            "low_stock": 0.5,
            "out_of_stock": 0,
            "pre_order": 0.3,
        }
        if "availability" in features_df.columns:
            features_df["availability_encoded"] = features_df["availability"].map(availability_map).fillna(0)
        
        return features_df
    
    def predict_viability(
        self,
        products: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Predict viability for products.
        
        Args:
            products: List of product dictionaries
            top_k: Return top K products by viability score
            
        Returns:
            List of viability predictions
        """
        if self.viability_model is None:
            raise ValueError("Viability model not loaded. Please train and save the model first.")
        
        # Convert to DataFrame
        df = pd.DataFrame(products)
        features_df = self.prepare_features(df)
        
        # Get feature columns expected by model
        feature_columns = self.viability_model.feature_names
        
        # Ensure all required features are present
        missing_features = set(feature_columns) - set(features_df.columns)
        if missing_features:
            # Fill missing features with 0
            for feat in missing_features:
                features_df[feat] = 0.0
        
        # Select and order features
        X = features_df[feature_columns]
        
        # Predict
        viability_scores = self.viability_model.predict_viability_score(X)
        
        # Get SHAP explanations if available
        try:
            explanations = self.viability_model.explain(X, return_shap_values=False)
            shap_available = True
        except Exception as e:
            logger.warning(f"Failed to get SHAP explanations: {e}")
            explanations = None
            shap_available = False
        
        # Format results
        results = []
        for i, product in enumerate(products):
            score = float(viability_scores[i])
            
            # Determine viability class
            if score >= 0.7:
                viability_class = "high"
            elif score >= 0.4:
                viability_class = "medium"
            else:
                viability_class = "low"
            
            result = {
                "sku": product.get("sku", f"product_{i}"),
                "viability_score": score,
                "viability_class": viability_class,
            }
            
            # Add SHAP values if available
            if shap_available and explanations:
                sample_explanation = explanations["per_sample_explanations"][i]
                result["shap_values"] = sample_explanation["feature_contributions"]
                result["base_value"] = sample_explanation["base_value"]
            else:
                result["shap_values"] = None
                result["base_value"] = None
            
            results.append(result)
        
        # Sort by viability score (descending)
        results.sort(key=lambda x: x["viability_score"], reverse=True)
        
        # Return top K if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def optimize_price(
        self,
        products: List[Dict[str, Any]],
        min_margin_percent: float = 0.15,
        enforce_map: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Optimize prices for products.
        
        Args:
            products: List of product dictionaries
            min_margin_percent: Minimum margin percentage
            enforce_map: Whether to enforce MAP constraints
            
        Returns:
            List of price optimizations
        """
        if self.price_optimizer is None:
            raise ValueError("Price optimizer not loaded. Please train and save the conversion model first.")
        
        # Update optimizer config
        self.price_optimizer.min_margin_percent = min_margin_percent
        self.price_optimizer.enforce_map = enforce_map
        
        # Optimize batch
        try:
            results = self.price_optimizer.optimize_batch(products)
        except Exception as e:
            logger.error(f"Price optimization failed: {e}")
            # Return default results
            results = []
            for product in products:
                landed_cost = (
                    product.get("cost", 0) +
                    product.get("shipping_cost", 0) +
                    product.get("duties", 0)
                )
                current_price = product.get("price", landed_cost * 1.5)
                results.append({
                    "sku": product.get("sku", "unknown"),
                    "current_price": current_price,
                    "recommended_price": current_price,
                    "expected_profit": 0.0,
                    "current_profit": 0.0,
                    "profit_improvement": 0.0,
                    "margin_percent": 0.0,
                    "conversion_probability": 0.0,
                    "map_constraint_applied": False,
                    "min_margin_constraint_applied": False,
                })
        
        return results
    
    def predict_stockout_risk(
        self,
        products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Predict stockout risk for products.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            List of stockout risk predictions
        """
        if self.stockout_model is None:
            raise ValueError("Stockout risk model not loaded. Please train and save the model first.")
        
        # Get feature columns expected by model
        feature_columns = self.stockout_model.feature_names
        
        # Batch predict
        results = self.stockout_model.predict_batch(products, feature_columns)
        
        # Add risk factors
        df = pd.DataFrame(products)
        features_df = self.prepare_features(df)
        
        # Ensure all required features are present
        missing_features = set(feature_columns) - set(features_df.columns)
        if missing_features:
            for feat in missing_features:
                features_df[feat] = 0.0
        
        X = features_df[feature_columns]
        risk_factors = self.stockout_model.get_risk_factors(X, products)
        
        # Combine results
        for i, result in enumerate(results):
            if i < len(risk_factors):
                result.update(risk_factors[i])
        
        return results
    
    def get_cluster_assignments(
        self,
        products: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Get cluster assignments for products.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            List of cluster IDs
        """
        if self.clusterer is None:
            # Return None cluster IDs if clustering not available
            return [None] * len(products)
        
        # Prepare texts
        texts = prepare_texts_for_clustering(products)
        
        # Predict clusters
        cluster_labels = self.clusterer.predict(texts)
        
        return cluster_labels.tolist()
    
    def process_complete_pipeline(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process complete pipeline: viability, price, risk, clustering.
        
        Args:
            df: Product DataFrame
            
        Returns:
            DataFrame with all predictions
        """
        products = df.to_dict("records")
        results_df = df.copy()
        
        # Viability predictions
        try:
            viability_results = self.predict_viability(products)
            viability_dict = {r["sku"]: r for r in viability_results}
            results_df["viability_score"] = results_df.get("sku", "").map(
                lambda x: viability_dict.get(x, {}).get("viability_score", 0.0)
            )
            results_df["viability_class"] = results_df.get("sku", "").map(
                lambda x: viability_dict.get(x, {}).get("viability_class", "low")
            )
        except Exception as e:
            logger.error(f"Failed to predict viability: {e}")
            results_df["viability_score"] = 0.0
            results_df["viability_class"] = "low"
        
        # Price optimization
        try:
            price_results = self.optimize_price(products)
            price_dict = {r["sku"]: r for r in price_results}
            results_df["recommended_price"] = results_df.get("sku", "").map(
                lambda x: price_dict.get(x, {}).get("recommended_price", results_df.loc[results_df.index[results_df.get("sku", "") == x], "price"].iloc[0] if len(results_df[results_df.get("sku", "") == x]) > 0 else 0.0)
            )
            results_df["expected_profit"] = results_df.get("sku", "").map(
                lambda x: price_dict.get(x, {}).get("expected_profit", 0.0)
            )
        except Exception as e:
            logger.error(f"Failed to optimize price: {e}")
            results_df["recommended_price"] = results_df.get("price", 0.0)
            results_df["expected_profit"] = 0.0
        
        # Stockout risk
        try:
            risk_results = self.predict_stockout_risk(products)
            risk_dict = {r["sku"]: r for r in risk_results}
            results_df["stockout_risk_score"] = results_df.get("sku", "").map(
                lambda x: risk_dict.get(x, {}).get("risk_score", 0.0)
            )
            results_df["stockout_risk_level"] = results_df.get("sku", "").map(
                lambda x: risk_dict.get(x, {}).get("risk_level", "low")
            )
        except Exception as e:
            logger.error(f"Failed to predict stockout risk: {e}")
            results_df["stockout_risk_score"] = 0.0
            results_df["stockout_risk_level"] = "low"
        
        # Clustering
        try:
            cluster_labels = self.get_cluster_assignments(products)
            results_df["cluster_id"] = cluster_labels
        except Exception as e:
            logger.error(f"Failed to get cluster assignments: {e}")
            results_df["cluster_id"] = None
        
        # Calculate margin
        results_df["landed_cost"] = (
            results_df.get("cost", 0) +
            results_df.get("shipping_cost", 0) +
            results_df.get("duties", 0)
        )
        results_df["margin_percent"] = (
            ((results_df["recommended_price"] - results_df["landed_cost"]) / results_df["recommended_price"]) * 100
        ).fillna(0)
        
        # Rank by viability
        results_df = results_df.sort_values("viability_score", ascending=False)
        results_df["rank"] = range(1, len(results_df) + 1)
        
        return results_df


# Global pipeline service instance
pipeline_service = MLPipelineService()

