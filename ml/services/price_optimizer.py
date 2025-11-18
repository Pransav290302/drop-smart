"""
Price Optimizer V3 — Compatible with ConversionModel V3
Works with LogisticRegression conversion model using FEATURES list.
"""

import numpy as np


class PriceOptimizer:
    def __init__(
        self,
        conversion_model,
        min_margin_percent: float = 0.15,
        enforce_map: bool = True,
        step: float = 0.10,
        max_change: float = 0.25,
    ):
        self.conversion_model = conversion_model
        self.min_margin_percent = min_margin_percent
        self.enforce_map = enforce_map
        self.step = step               # price step = +0.10 USD
        self.max_change = max_change   # max 25% price increase

    # -----------------------------------------------------------
    # CORE HELPERS
    # -----------------------------------------------------------

    def compute_margin_percent(self, price, landed_cost):
        if price <= 0:
            return 0
        return max(0, (price - landed_cost) / price)

    def compute_profit(self, price, landed_cost, conversion_prob):
        margin = self.compute_margin_percent(price, landed_cost)
        return margin * conversion_prob * price

    # -----------------------------------------------------------
    # PRICE OPTIMIZATION FOR A SINGLE PRODUCT
    # -----------------------------------------------------------

    def optimize_single(self, product: dict) -> dict:

        # Basic fields
        cost = float(product.get("cost", 0))
        shipping = float(product.get("shipping_cost", 0))
        duties = float(product.get("duties", 0))
        current_price = float(product.get("price", 0))

        landed_cost = cost + shipping + duties
        map_price = float(product.get("map_price", 0)) if self.enforce_map else 0

        # Allowed price range
        min_price = max(
            landed_cost * (1 + self.min_margin_percent),  # min margin constraint
            map_price,                                    # MAP constraint
        )

        max_price = current_price * (1 + self.max_change)  # e.g., +25%
        if max_price <= min_price:
            max_price = min_price + self.step

        # Candidate price grid
        candidate_prices = np.arange(min_price, max_price, self.step)

        best_price = current_price
        best_profit = -1
        best_conversion_prob = 0

        for price in candidate_prices:

            # ⭕ Call ConversionModel V3 prediction function
            conv_prob = self.conversion_model.predict_conversion_probability(
                product=product,
                new_price=price
            )

            profit = self.compute_profit(price, landed_cost, conv_prob)

            if profit > best_profit:
                best_profit = profit
                best_price = price
                best_conversion_prob = conv_prob

        result = {
            "sku": product.get("sku", "unknown"),
            "current_price": current_price,
            "recommended_price": round(best_price, 2),
            "conversion_probability": round(best_conversion_prob, 4),
            "expected_profit": round(best_profit, 4),
            "current_profit": round(
                self.compute_profit(current_price, landed_cost,
                                    self.conversion_model.predict_conversion_probability(product, current_price)), 4
            ),
        }

        result["profit_improvement"] = round(
            result["expected_profit"] - result["current_profit"], 4
        )

        result["margin_percent"] = round(
            self.compute_margin_percent(best_price, landed_cost) * 100, 2
        )

        # Constraint flags
        result["map_constraint_applied"] = best_price <= map_price if self.enforce_map else False
        result["min_margin_constraint_applied"] = best_price <= landed_cost * (1 + self.min_margin_percent)

        return result

    # -----------------------------------------------------------
    # BATCH PROCESSING
    # -----------------------------------------------------------

    def optimize_batch(self, products: list) -> list:
        results = []
        for product in products:
            try:
                results.append(self.optimize_single(product))
            except Exception as e:
                results.append({
                    "sku": product.get("sku", "unknown"),
                    "error": str(e),
                    "current_price": product.get("price", 0),
                    "recommended_price": product.get("price", 0),
                    "expected_profit": 0,
                    "current_profit": 0,
                    "profit_improvement": 0,
                    "conversion_probability": 0,
                    "margin_percent": 0,
                    "map_constraint_applied": False,
                    "min_margin_constraint_applied": False,
                })
        return results
