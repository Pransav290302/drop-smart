"""Data normalization module - Normalize currencies, dimensions, and weights"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from ml.config import get_schema_config

logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    Normalizes currencies, dimensions, and weights according to schema configuration.
    
    FR-3: Normalize currencies, dimensions, and weights.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data normalizer.
        
        Args:
            config: Normalization configuration (if None, loads from config file)
        """
        if config is None:
            schema_config = get_schema_config()
            config = schema_config
        
        self.config = config
        
        # Currency conversion rates
        currency_config = config.get("currency", {})
        self.default_currency = currency_config.get("default", "USD")
        self.conversion_rates = currency_config.get("conversion_rates", {"USD": 1.0})
        
        # Unit configurations
        units_config = config.get("units", {})
        self.default_weight_unit = units_config.get("weight", {}).get("default", "kg")
        self.default_dimension_unit = units_config.get("dimension", {}).get("default", "cm")
        
        logger.info(f"Initialized DataNormalizer with default currency: {self.default_currency}")
    
    def normalize_currency(
        self,
        df: pd.DataFrame,
        amount_columns: List[str],
        currency_column: Optional[str] = None,
        target_currency: str = "USD"
    ) -> pd.DataFrame:
        """
        Normalize currency values to target currency.
        
        Args:
            df: DataFrame with currency values
            amount_columns: List of column names containing amounts
            currency_column: Column name indicating currency (if None, assumes default)
            target_currency: Target currency for conversion (default: USD)
            
        Returns:
            DataFrame with normalized currency values
        """
        df_normalized = df.copy()
        
        # If no currency column, assume all values are in default currency
        if currency_column is None or currency_column not in df.columns:
            # Check if values need conversion (if they're already in target, no conversion needed)
            logger.info(f"Assuming all amounts are in {self.default_currency}")
            source_currency = self.default_currency
        else:
            # Get unique currencies in the data
            unique_currencies = df[currency_column].unique()
            source_currency = unique_currencies[0] if len(unique_currencies) == 1 else None
        
        # Convert each amount column
        for col in amount_columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping currency normalization")
                continue
            
            # Get conversion rate
            if currency_column and currency_column in df.columns:
                # Per-row conversion based on currency column
                def convert_row(row):
                    source_curr = str(row.get(currency_column, self.default_currency)).upper()
                    amount = row.get(col, 0.0)
                    
                    if pd.isna(amount) or amount == 0:
                        return amount
                    
                    # Get conversion rates
                    source_rate = self.conversion_rates.get(source_curr, 1.0)
                    target_rate = self.conversion_rates.get(target_currency.upper(), 1.0)
                    
                    # Convert: target_amount = source_amount * (target_rate / source_rate)
                    if source_rate > 0:
                        converted = amount * (target_rate / source_rate)
                        return converted
                    return amount
                
                df_normalized[col] = df.apply(convert_row, axis=1)
            else:
                # Single currency conversion
                source_rate = self.conversion_rates.get(source_currency.upper(), 1.0)
                target_rate = self.conversion_rates.get(target_currency.upper(), 1.0)
                
                if source_rate > 0 and source_rate != target_rate:
                    conversion_factor = target_rate / source_rate
                    df_normalized[col] = df[col] * conversion_factor
                    logger.info(f"Converted {col} from {source_currency} to {target_currency} (factor: {conversion_factor:.4f})")
        
        return df_normalized
    
    def normalize_weight(
        self,
        df: pd.DataFrame,
        weight_column: str,
        weight_unit_column: Optional[str] = None,
        target_unit: str = "kg"
    ) -> pd.DataFrame:
        """
        Normalize weight values to target unit.
        
        Supported units: kg, g, lb, oz
        
        Args:
            df: DataFrame with weight values
            weight_column: Column name containing weight
            weight_unit_column: Column name indicating unit (if None, assumes default)
            target_unit: Target unit for conversion (default: kg)
            
        Returns:
            DataFrame with normalized weight values
        """
        df_normalized = df.copy()
        
        if weight_column not in df.columns:
            logger.warning(f"Weight column '{weight_column}' not found")
            return df_normalized
        
        # Conversion factors to kg
        to_kg = {
            "kg": 1.0,
            "g": 0.001,
            "lb": 0.453592,
            "oz": 0.0283495,
        }
        
        # Conversion from kg to target
        from_kg = {
            "kg": 1.0,
            "g": 1000.0,
            "lb": 2.20462,
            "oz": 35.274,
        }
        
        if weight_unit_column and weight_unit_column in df.columns:
            # Per-row conversion based on unit column
            def convert_weight(row):
                source_unit = str(row.get(weight_unit_column, self.default_weight_unit)).lower()
                weight = row.get(weight_column, 0.0)
                
                if pd.isna(weight) or weight == 0:
                    return weight
                
                # Convert to kg first, then to target
                source_to_kg = to_kg.get(source_unit, 1.0)
                kg_to_target = from_kg.get(target_unit.lower(), 1.0)
                
                weight_kg = weight * source_to_kg
                weight_target = weight_kg * kg_to_target
                
                return weight_target
            
            df_normalized[weight_column] = df.apply(convert_weight, axis=1)
        else:
            # Assume all weights are in default unit
            source_unit = self.default_weight_unit.lower()
            source_to_kg = to_kg.get(source_unit, 1.0)
            kg_to_target = from_kg.get(target_unit.lower(), 1.0)
            
            if source_to_kg != 1.0 or kg_to_target != 1.0:
                conversion_factor = source_to_kg * kg_to_target
                df_normalized[weight_column] = df[weight_column] * conversion_factor
                logger.info(f"Converted {weight_column} from {source_unit} to {target_unit}")
        
        return df_normalized
    
    def normalize_dimension(
        self,
        df: pd.DataFrame,
        dimension_columns: List[str],
        dimension_unit_column: Optional[str] = None,
        target_unit: str = "cm"
    ) -> pd.DataFrame:
        """
        Normalize dimension values to target unit.
        
        Supported units: cm, m, in, ft
        
        Args:
            df: DataFrame with dimension values
            dimension_columns: List of column names containing dimensions
            dimension_unit_column: Column name indicating unit (if None, assumes default)
            target_unit: Target unit for conversion (default: cm)
            
        Returns:
            DataFrame with normalized dimension values
        """
        df_normalized = df.copy()
        
        # Conversion factors to cm
        to_cm = {
            "cm": 1.0,
            "m": 100.0,
            "in": 2.54,
            "ft": 30.48,
        }
        
        # Conversion from cm to target
        from_cm = {
            "cm": 1.0,
            "m": 0.01,
            "in": 0.393701,
            "ft": 0.0328084,
        }
        
        for col in dimension_columns:
            if col not in df.columns:
                logger.warning(f"Dimension column '{col}' not found, skipping")
                continue
            
            if dimension_unit_column and dimension_unit_column in df.columns:
                # Per-row conversion
                def convert_dimension(row):
                    source_unit = str(row.get(dimension_unit_column, self.default_dimension_unit)).lower()
                    dimension = row.get(col, 0.0)
                    
                    if pd.isna(dimension) or dimension == 0:
                        return dimension
                    
                    # Convert to cm first, then to target
                    source_to_cm = to_cm.get(source_unit, 1.0)
                    cm_to_target = from_cm.get(target_unit.lower(), 1.0)
                    
                    dimension_cm = dimension * source_to_cm
                    dimension_target = dimension_cm * cm_to_target
                    
                    return dimension_target
                
                df_normalized[col] = df.apply(convert_dimension, axis=1)
            else:
                # Assume all dimensions are in default unit
                source_unit = self.default_dimension_unit.lower()
                source_to_cm = to_cm.get(source_unit, 1.0)
                cm_to_target = from_cm.get(target_unit.lower(), 1.0)
                
                if source_to_cm != 1.0 or cm_to_target != 1.0:
                    conversion_factor = source_to_cm * cm_to_target
                    df_normalized[col] = df[col] * conversion_factor
                    logger.info(f"Converted {col} from {source_unit} to {target_unit}")
        
        return df_normalized
    
    def normalize_all(
        self,
        df: pd.DataFrame,
        currency_columns: Optional[List[str]] = None,
        weight_column: Optional[str] = None,
        dimension_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalize all currency, weight, and dimension columns.
        
        Args:
            df: DataFrame to normalize
            currency_columns: List of currency column names (default: cost, price, shipping_cost, duties, map_price)
            weight_column: Weight column name (default: weight_kg)
            dimension_columns: List of dimension column names (default: length_cm, width_cm, height_cm)
            
        Returns:
            DataFrame with all normalized values
        """
        df_normalized = df.copy()
        
        # Default currency columns
        if currency_columns is None:
            currency_columns = ["cost", "price", "shipping_cost", "duties", "map_price"]
            currency_columns = [col for col in currency_columns if col in df.columns]
        
        # Normalize currencies
        if currency_columns:
            df_normalized = self.normalize_currency(
                df_normalized,
                currency_columns,
                target_currency=self.default_currency
            )
        
        # Normalize weight
        if weight_column:
            if weight_column in df.columns:
                df_normalized = self.normalize_weight(
                    df_normalized,
                    weight_column,
                    target_unit=self.default_weight_unit
                )
        
        # Default dimension columns
        if dimension_columns is None:
            dimension_columns = ["length_cm", "width_cm", "height_cm"]
            dimension_columns = [col for col in dimension_columns if col in df.columns]
        
        # Normalize dimensions
        if dimension_columns:
            df_normalized = self.normalize_dimension(
                df_normalized,
                dimension_columns,
                target_unit=self.default_dimension_unit
            )
        
        logger.info("Data normalization complete")
        
        return df_normalized


def normalize_dataframe(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Convenience function to normalize a DataFrame.
    
    Args:
        df: DataFrame to normalize
        config: Optional normalization configuration
        
    Returns:
        Normalized DataFrame
    """
    normalizer = DataNormalizer(config)
    return normalizer.normalize_all(df)

