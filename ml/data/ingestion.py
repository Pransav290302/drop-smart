"""Excel file ingestion module"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_excel_file(file_path: Path) -> pd.DataFrame:
    """
    Load Excel file into pandas DataFrame
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        DataFrame with product data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be read
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Try reading Excel file
        df = pd.read_excel(file_path, engine='openpyxl')
        logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise ValueError(f"Failed to load Excel file: {e}")


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Basic DataFrame validation
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If DataFrame is invalid
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if len(df) == 0:
        raise ValueError("DataFrame has no rows")
    
    return True

