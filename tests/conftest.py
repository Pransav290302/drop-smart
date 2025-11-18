"""Pytest configuration and fixtures"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing"""
    data = {
        "sku": ["SKU001", "SKU002", "SKU003"],
        "product_name": ["Product 1", "Product 2", "Product 3"],
        "cost": [10.0, 20.0, 30.0],
        "price": [15.0, 30.0, 45.0],
        "shipping_cost": [2.0, 3.0, 4.0],
        "lead_time_days": [7, 14, 21],
        "availability": ["in_stock", "low_stock", "in_stock"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_excel_file(temp_dir, sample_dataframe):
    """Create sample Excel file for testing"""
    file_path = temp_dir / "sample.xlsx"
    sample_dataframe.to_excel(file_path, index=False, engine='openpyxl')
    return file_path

