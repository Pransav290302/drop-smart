"""
Price Optimization Service — Optimizes Expected Profit with Constraints
Compatible with new ConversionModel + Viability + Stockout pipelines
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

from ml.models.price_model import ConversionModel
from ml.config import get_model_config

logger = logging.getLogger(__name__)


class PriceOptimizer:
    """
    Price optimizer:
        argmax_price  p(price | features) × (price – landed_cost)

    Implements:
        • Candidate price generation
        • Conversion probability per price
        • Expected profit estimation
        • MAP + Min-margin enforcement
    """

    def __init__(
        self,
        conversion_model: Optional[ConversionModel] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.conversion_model = conversion_model

        # Load config if not provided
        if config is None:
            model_config = get_model_config()
            config = model_config.get("price_optimizer", {})

        self.config = config
        self.price_range_multiplier = config.get("price_range_multiplier", [0.8, 2.0])
        self.price_step = config.get("price_step", 0.05)
        self.min_margin_percent = config.get("min_margin_percent", 0.15)
        self.enforce_map = config.get("enforce_map", True)

    # ---------------------------------------------------------
    # BASIC CALCULATIONS
    # ---------------------------------------------------------
    def calculate_landed_cost(self, product: Dict[str, Any]) -> float:
        return (
            product.get("cost", 0) +
            product.get("shipping_cost", 0) +
            product.get("duties", 0)
        )

    def calculate_margin_percent(self, price: float, landed_cost: float) -> float:
        if price <= 0:
            return 0.0
        return ((price - landed_cost) / price) * 100

    # ---------------------------------------------------------
    # PRICE RANGE GENERATION
    # ---------------------------------------------------------
    def generate_candidate_prices(
        self,
        landed_cost: float,
        current_price: Optional[float] = None,
        map_price: Optional[float] = None
    ) -> np.ndarray:

        min_price = landed_cost * self.price_range_multiplier[0]
        max_price = landed_cost * self.price_range_multiplier[1]

        # Apply MAP constraint
        if self.enforce_map and map_price is not None:
            max_price = min(max_price, map_price)

        # Enforce minimum margin
        min_price_with_margin = landed_cost / (1 - self.min_margin_percent / 100)
        min_price = max(min_price, min_price_with_margin)

        # Generate price grid
        num_steps = int((max_price - min_price) / self.price_step) + 1
        if num_steps < 1:
            num_steps = 1

        prices = np.linspace(min_price, max_price, num_steps)
        prices = np.round(prices, 2)

        # Add current price if valid
        if current_price and min_price <= current_price <= max_price:
            prices = np.append(prices, round(current_price, 2))
            prices = np.unique(prices)

        return np.sort(prices)

    # ---------------------------------------------------------
    # MAIN OPTIMIZATION
    # ---------------------------------------------------------
    def optimize_price(
        self,
        product: Dict[str, Any],
        features: Optional[pd.DataFrame] = None,
        current_price: Optional[float] = None
    ) -> Dict[str, Any]:

        if self.conversion_model is None:
            raise ValueError("Conversion model must be set before optimization.")

        if not self.conversion_model.is_trained:
            raise ValueError("Conversion model must be trained first.")

        landed_cost = self.calculate_landed_cost(product)
        if landed_cost <= 0:
            raise ValueError("Invalid landed cost.")

        if current_price is None:
            current_price = product.get("price", landed_cost * 1.5)

        map_price = product.get("map_price")

        # Build correct feature matrix
        if features is None:
            features = self._extract_features(product)

        candidate_prices = self.generate_candidate_prices(
            landed_cost, current_price, map_price
        )

        best_price = current_price
        best_profit = 0.0
        best_conv = 0.0

        for price in candidate_prices:
            try:
                conv_prob = self.conversion_model.predict_for_price(
                    price,
                    features,
                    price_feature=self.conversion_model.price_feature_name
                )
            except Exception as e:
                logger.warning(f"Conversion model failed at price {price}: {e}")
                continue

            profit = conv_prob * (price - landed_cost)

            if profit > best_profit:
                best_profit = profit
                best_price = price
                best_conv = conv_prob

        # Apply MAP / margin rules
        best_price, constraints = self._apply_constraints(
            best_price, landed_cost, map_price
        )

        # Recompute after constraints
        final_conv = self.conversion_model.predict_for_price(
            best_price, features, price_feature=self.conversion_model.price_feature_name
        )
        final_profit = final_conv * (best_price - landed_cost)

        # Compute current profit
        curr_conv = self.conversion_model.predict_for_price(
            current_price, features, price_feature=self.conversion_model.price_feature_name
        )
        curr_profit = curr_conv * (current_price - landed_cost)

        if curr_profit > 0:
            improvement = ((final_profit - curr_profit) / curr_profit) * 100
        else:
            improvement = 100.0 if final_profit > 0 else 0.0

        return {
            "recommended_price": round(best_price, 2),
            "expected_profit": round(final_profit, 2),
            "conversion_probability": round(final_conv, 4),
            "margin_percent": round(self.calculate_margin_percent(best_price, landed_cost), 2),

            "current_price": round(current_price, 2),
            "current_profit": round(curr_profit, 2),
            "profit_improvement": round(improvement, 2),

            "map_constraint_applied": constraints["map"],
            "min_margin_constraint_applied": constraints["min_margin"],
        }

    # ---------------------------------------------------------
    # FEATURE EXTRACTION (CRITICAL PART)
    # MUST MATCH TRAINING FEATURES EXACTLY
    # ---------------------------------------------------------
    def _extract_features(self, product: Dict[str, Any]) -> pd.DataFrame:
        """
        Build feature vector EXACTLY matching training pipeline:

        FEATURES = [
            "price", "cost", "shipping_cost", "duties",
            "lead_time_days", "stock", "inventory", "quantity",
            "demand", "past_sales",
            "weight_kg", "length_cm", "width_cm", "height_cm",
            "margin", "supplier_reliability_score"
        ]
        """

        cost = product.get("cost", 0)
        shipping = product.get("shipping_cost", 0)
        duties = product.get("duties", 0)
        price = product.get("price", cost * 1.5)

        landed_cost = cost + shipping + duties
        margin = 0
        if price > 0:
            margin = max((price - landed_cost) / price, 0)

        features = {
            "price": price,
            "cost": cost,
            "shipping_cost": shipping,
            "duties": duties,

            "lead_time_days": product.get("lead_time_days", 14),
            "stock": product.get("stock", 100),
            "inventory": product.get("inventory", 100),
            "quantity": product.get("quantity", 1),
            "demand": product.get("demand", 50),
            "past_sales": product.get("past_sales", 20),

            "weight_kg": product.get("weight_kg", 0.5),
            "length_cm": product.get("length_cm", 10),
            "width_cm": product.get("width_cm", 5),
            "height_cm": product.get("height_cm", 3),

            "margin": margin,
            "supplier_reliability_score": product.get("supplier_reliability_score", 0.8)
        }

        return pd.DataFrame([features])

    # ---------------------------------------------------------
    # CONSTRAINTS
    # ---------------------------------------------------------
    def _apply_constraints(
        self,
        price: float,
        landed_cost: float,
        map_price: Optional[float]
    ) -> Tuple[float, Dict[str, bool]]:

        constraints = {"map": False, "min_margin": False}
        final_price = price

        # MAP
        if self.enforce_map and map_price is not None:
            if final_price > map_price:
                final_price = map_price
                constraints["map"] = True

        # Min margin
        min_price_allowed = landed_cost / (1 - self.min_margin_percent / 100)
        if final_price < min_price_allowed:
            final_price = min_price_allowed
            constraints["min_margin"] = True

        return final_price, constraints

    # ---------------------------------------------------------
    # BATCH OPTIMIZATION
    # ---------------------------------------------------------
    def optimize_batch(
        self,
        products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        results = []
        for prod in products:
            try:
                r = self.optimize_price(prod)
                r["sku"] = prod.get("sku")
                results.append(r)
            except Exception as e:
                logger.error(f"Failed to optimize {prod.get('sku')}: {e}")
        return results

    def set_conversion_model(self, conversion_model: ConversionModel) -> None:
        if not conversion_model.is_trained:
            raise ValueError("Conversion model must be trained before use.")
        self.conversion_model = conversion_model
        logger.info("Conversion model attached successfully")
