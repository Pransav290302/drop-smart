"""Main feature engineering module"""

import pandas as pd
from typing import Dict, Any, Optional
import logging

from ml.features.weight_features import add_weight_features
from ml.features.time_features import add_time_features

logger = logging.getLogger(__name__)


def engineer_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Main feature engineering function
    
    Orchestrates all feature engineering steps:
    - Cost features (landed cost, margin)
    - Weight features (volumetric weight, size tier)
    - Time features (lead-time buckets, seasonality)
    - Embeddings (product title embeddings - handled separately in clustering)
    
    FR-4: Compute derived fields:
    - Landed cost = cost + shipping + duties
    - Margin %
    - Volumetric weight & size tier
    - Lead-time buckets
    - Seasonality indicator
    
    Args:
        df: Raw DataFrame
        config: Optional configuration dictionary
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Starting feature engineering...")
    
    # Create a copy to avoid modifying original
    df_features = df.copy()
    
    # 1. Cost features (landed cost, margin)
    # Calculate landed cost
    cost_cols = ["cost", "shipping_cost", "duties"]
    available_cost_cols = [col for col in cost_cols if col in df_features.columns]
    
    if available_cost_cols:
        df_features["landed_cost"] = df_features[available_cost_cols].sum(axis=1, skipna=True).fillna(0)
    else:
        df_features["landed_cost"] = 0.0
    
    # Calculate margin percent
    if "price" in df_features.columns and "landed_cost" in df_features.columns:
        df_features["margin_percent"] = (
            ((df_features["price"] - df_features["landed_cost"]) / df_features["price"]) * 100
        ).fillna(0)
    else:
        df_features["margin_percent"] = 0.0
    
    # 2. Weight features (volumetric weight, size tier)
    df_features = add_weight_features(
        df_features,
        length_col="length_cm",
        width_col="width_cm",
        height_col="height_cm",
        weight_col="weight_kg",
        volumetric_divisor=config.get("volumetric_weight_divisor", 5000.0) if config else 5000.0
    )
    
    # 3. Time features (lead-time buckets, seasonality)
    lead_time_buckets = None
    if config and "lead_time_buckets" in config:
        lead_time_buckets = config["lead_time_buckets"]
    
    df_features = add_time_features(
        df_features,
        lead_time_col="lead_time_days",
        category_col="category",
        lead_time_buckets=lead_time_buckets
    )
    
    # 4. Embeddings are handled separately in clustering module
    # Product title embeddings are generated during clustering, not in feature engineering
    
    logger.info(f"Feature engineering complete. Shape: {df_features.shape}")
    logger.info(f"Added features: landed_cost, margin_percent, volumetric_weight, size_tier, "
                f"lead_time_bucket, lead_time_category, seasonality indicators")
    
    return df_features
