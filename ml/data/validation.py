"""Data validation module"""

import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from ml.config import get_schema_config

logger = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate DataFrame against required schema
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    schema_config = get_schema_config()
    required_fields = schema_config.get("required_fields", [])
    
    errors = []
    warnings = []
    
    # Check required fields
    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Check optional fields
    optional_fields = schema_config.get("optional_fields", [])
    missing_optional = [field for field in optional_fields if field not in df.columns]
    if missing_optional:
        warnings.append(f"Missing optional fields: {', '.join(missing_optional)}")
    
    is_valid = len(errors) == 0
    
    return is_valid, errors, {"warnings": warnings}


def validate_field_types(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate field types according to schema
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, errors)
    """
    schema_config = get_schema_config()
    field_validation = schema_config.get("field_validation", {})
    
    errors = []
    
    for field, rules in field_validation.items():
        if field not in df.columns:
            continue
        
        field_type = rules.get("type")
        if field_type == "float":
            if not pd.api.types.is_numeric_dtype(df[field]):
                errors.append(f"Field '{field}' should be numeric")
        elif field_type == "integer":
            if not pd.api.types.is_integer_dtype(df[field]):
                errors.append(f"Field '{field}' should be integer")
        elif field_type == "string":
            if not pd.api.types.is_string_dtype(df[field]):
                errors.append(f"Field '{field}' should be string")
    
    is_valid = len(errors) == 0
    return is_valid, errors

