"""Pydantic schemas for API requests and responses (FULLY UPDATED FOR V3 PIPELINE)
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


# ---------------------------------------------------------
# ENUMS
# ---------------------------------------------------------

class AvailabilityStatus(str, Enum):
    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    PRE_ORDER = "pre_order"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------
# PRODUCT INPUT SCHEMA (FULL FEATURE SET)
# ---------------------------------------------------------

class ProductInput(BaseModel):
    """
    Full product schema used by all ML models.

    Compatible with training pipeline V3:
    FEATURES = [
        price, cost, shipping_cost, duties,
        lead_time_days, stock, inventory, quantity,
        demand, past_sales,
        weight_kg, length_cm, width_cm, height_cm,
        margin, supplier_reliability_score
    ]
    """

    # Required
    sku: str
    product_name: str
    cost: float
    price: float
    shipping_cost: float
    lead_time_days: int
    availability: AvailabilityStatus

    # Optional but used by ML models
    duties: Optional[float] = 0.0

    stock: Optional[float] = 0.0
    inventory: Optional[float] = 0.0
    quantity: Optional[float] = 0.0
    demand: Optional[float] = 0.0
    past_sales: Optional[float] = 0.0

    weight_kg: Optional[float] = 0.0
    length_cm: Optional[float] = 0.0
    width_cm: Optional[float] = 0.0
    height_cm: Optional[float] = 0.0

    supplier_reliability_score: Optional[float] = 0.5  # middle default

    # Optional text fields for clustering
    description: Optional[str] = None
    category: Optional[str] = None
    title: Optional[str] = None
    name: Optional[str] = None

    # MAP price rules
    map_price: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "sku": "ABC123",
                "product_name": "Gaming Mouse",
                "cost": 20,
                "price": 49.99,
                "shipping_cost": 5,
                "duties": 1.2,
                "lead_time_days": 7,
                "availability": "in_stock",
                "stock": 50,
                "inventory": 100,
                "quantity": 1,
                "demand": 30,
                "past_sales": 15,
                "weight_kg": 0.3,
                "length_cm": 12,
                "width_cm": 6,
                "height_cm": 4,
                "supplier_reliability_score": 0.8,
                "category": "Electronics",
                "description": "High precision wired gaming mouse"
            }
        }


# ---------------------------------------------------------
# BULK PRODUCT INPUT
# ---------------------------------------------------------

class BulkProductInput(BaseModel):
    products: List[ProductInput]

    @validator("products")
    def validate_count(cls, v):
        if len(v) == 0:
            raise ValueError("At least 1 product required")
        if len(v) > 5000:
            raise ValueError("Max 5000 products allowed")
        return v


# ---------------------------------------------------------
# REQUEST SCHEMAS
# ---------------------------------------------------------

class ValidationRequest(BaseModel):
    file_id: str


class PredictViabilityRequest(BaseModel):
    products: List[ProductInput]


class OptimizePriceRequest(BaseModel):
    products: List[ProductInput]
    min_margin_percent: Optional[float] = Field(0.15, ge=0, le=1)
    enforce_map: Optional[bool] = True


class StockoutRiskRequest(BaseModel):
    products: List[ProductInput]


# ---------------------------------------------------------
# RESPONSE SCHEMAS
# ---------------------------------------------------------

class ValidationError(BaseModel):
    field: str
    message: str
    row: Optional[int] = None


class ValidationResponse(BaseModel):
    is_valid: bool
    errors: List[ValidationError] = []
    warnings: List[str] = []
    total_rows: int
    total_columns: int
    missing_required_fields: List[str] = []
    missing_optional_fields: List[str] = []


class ViabilityPrediction(BaseModel):
    sku: str
    viability_score: float
    viability_class: str
    shap_values: Optional[Dict[str, float]] = None


class ViabilityResponse(BaseModel):
    predictions: List[ViabilityPrediction]
    model_version: str
    processing_time_seconds: float


class PriceOptimization(BaseModel):
    sku: str
    current_price: float
    recommended_price: float
    expected_profit: float
    current_profit: float
    profit_improvement: float
    margin_percent: float
    conversion_probability: float
    map_constraint_applied: bool
    min_margin_constraint_applied: bool


class PriceOptimizationResponse(BaseModel):
    optimizations: List[PriceOptimization]
    model_version: str
    processing_time_seconds: float


class StockoutRiskPrediction(BaseModel):
    sku: str
    risk_score: float
    risk_level: RiskLevel
    risk_factors: List[str] = []
    lead_time_risk: bool = False
    availability_risk: bool = False


class StockoutRiskResponse(BaseModel):
    predictions: List[StockoutRiskPrediction]
    model_version: str
    processing_time_seconds: float


class ProductResult(BaseModel):
    sku: str
    product_name: str
    viability_score: float
    viability_class: str
    recommended_price: float
    current_price: float
    margin_percent: float
    stockout_risk_score: float
    stockout_risk_level: RiskLevel
    cluster_id: Optional[int]
    rank: int


class ResultsResponse(BaseModel):
    results: List[ProductResult]
    total_products: int
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    model_versions: Dict[str, str]


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    file_size_bytes: int
    total_rows: int
    upload_timestamp: datetime = Field(default_factory=datetime.now)
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
