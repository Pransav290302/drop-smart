"""Pydantic schemas for API requests and responses"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class AvailabilityStatus(str, Enum):
    """Product availability status"""
    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    PRE_ORDER = "pre_order"


class RiskLevel(str, Enum):
    """Stockout risk level"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Request Schemas

class ProductInput(BaseModel):
    """Input product data"""
    sku: str = Field(..., description="Product SKU")
    product_name: str = Field(..., description="Product name")
    cost: float = Field(..., gt=0, description="Product cost")
    price: float = Field(..., gt=0, description="Current price")
    shipping_cost: float = Field(..., ge=0, description="Shipping cost")
    lead_time_days: int = Field(..., ge=0, le=365, description="Lead time in days")
    availability: AvailabilityStatus = Field(..., description="Availability status")
    description: Optional[str] = Field(None, description="Product description")
    category: Optional[str] = Field(None, description="Product category")
    weight_kg: Optional[float] = Field(None, ge=0, description="Weight in kg")
    length_cm: Optional[float] = Field(None, ge=0, description="Length in cm")
    width_cm: Optional[float] = Field(None, ge=0, description="Width in cm")
    height_cm: Optional[float] = Field(None, ge=0, description="Height in cm")
    map_price: Optional[float] = Field(None, ge=0, description="Minimum Advertised Price")
    duties: Optional[float] = Field(None, ge=0, description="Duties cost")
    supplier_name: Optional[str] = Field(None, description="Supplier name")
    supplier_reliability_score: Optional[float] = Field(None, ge=0, le=1, description="Supplier reliability score")

    class Config:
        json_schema_extra = {
            "example": {
                "sku": "SKU001",
                "product_name": "Wireless Headphones",
                "cost": 25.0,
                "price": 49.99,
                "shipping_cost": 5.0,
                "lead_time_days": 7,
                "availability": "in_stock",
                "description": "High-quality wireless headphones",
                "category": "Electronics",
                "weight_kg": 0.3,
                "length_cm": 20.0,
                "width_cm": 15.0,
                "height_cm": 8.0,
                "map_price": 45.0,
                "duties": 2.5,
            }
        }


class BulkProductInput(BaseModel):
    """Bulk product input for batch processing"""
    products: List[ProductInput] = Field(..., description="List of products")

    @validator("products")
    def validate_product_count(cls, v):
        if len(v) > 10000:
            raise ValueError("Maximum 10000 products allowed per request")
        if len(v) == 0:
            raise ValueError("At least one product is required")
        return v


class ValidationRequest(BaseModel):
    """Request for schema validation"""
    file_id: str = Field(..., description="Uploaded file ID")


class PredictViabilityRequest(BaseModel):
    """Request for viability prediction"""
    products: List[ProductInput] = Field(..., description="List of products to predict")


class OptimizePriceRequest(BaseModel):
    """Request for price optimization"""
    products: List[ProductInput] = Field(..., description="List of products to optimize")
    min_margin_percent: Optional[float] = Field(0.15, ge=0, le=1, description="Minimum margin percentage")
    enforce_map: Optional[bool] = Field(True, description="Enforce MAP constraints")


class StockoutRiskRequest(BaseModel):
    """Request for stockout risk prediction"""
    products: List[ProductInput] = Field(..., description="List of products to analyze")


# Response Schemas

class ValidationError(BaseModel):
    """Validation error detail"""
    field: str = Field(..., description="Field name")
    message: str = Field(..., description="Error message")
    row: Optional[int] = Field(None, description="Row number if applicable")


class ValidationResponse(BaseModel):
    """Schema validation response"""
    is_valid: bool = Field(..., description="Whether schema is valid")
    errors: List[ValidationError] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    total_rows: int = Field(..., description="Total number of rows")
    total_columns: int = Field(..., description="Total number of columns")
    missing_required_fields: List[str] = Field(default_factory=list, description="Missing required fields")
    missing_optional_fields: List[str] = Field(default_factory=list, description="Missing optional fields")


class ViabilityPrediction(BaseModel):
    """Viability prediction result"""
    sku: str = Field(..., description="Product SKU")
    viability_score: float = Field(..., ge=0, le=1, description="Probability of sale within 30 days")
    viability_class: str = Field(..., description="Viability class (high/medium/low)")
    shap_values: Optional[Dict[str, float]] = Field(None, description="SHAP feature importance values")


class ViabilityResponse(BaseModel):
    """Viability prediction response"""
    predictions: List[ViabilityPrediction] = Field(..., description="List of predictions")
    model_version: str = Field(..., description="Model version used")
    processing_time_seconds: float = Field(..., description="Processing time")


class PriceOptimization(BaseModel):
    """Price optimization result"""
    sku: str = Field(..., description="Product SKU")
    current_price: float = Field(..., description="Current price")
    recommended_price: float = Field(..., description="Recommended optimized price")
    expected_profit: float = Field(..., description="Expected profit at recommended price")
    current_profit: float = Field(..., description="Current profit")
    profit_improvement: float = Field(..., description="Profit improvement percentage")
    margin_percent: float = Field(..., description="Margin percentage at recommended price")
    conversion_probability: float = Field(..., ge=0, le=1, description="Conversion probability at recommended price")
    map_constraint_applied: bool = Field(..., description="Whether MAP constraint was applied")
    min_margin_constraint_applied: bool = Field(..., description="Whether min margin constraint was applied")


class PriceOptimizationResponse(BaseModel):
    """Price optimization response"""
    optimizations: List[PriceOptimization] = Field(..., description="List of optimizations")
    model_version: str = Field(..., description="Model version used")
    processing_time_seconds: float = Field(..., description="Processing time")


class StockoutRiskPrediction(BaseModel):
    """Stockout risk prediction result"""
    sku: str = Field(..., description="Product SKU")
    risk_score: float = Field(..., ge=0, le=1, description="Risk probability score")
    risk_level: RiskLevel = Field(..., description="Risk level")
    risk_factors: List[str] = Field(default_factory=list, description="List of risk factors")
    lead_time_risk: bool = Field(..., description="Whether lead time is a risk factor")
    availability_risk: bool = Field(..., description="Whether availability is a risk factor")


class StockoutRiskResponse(BaseModel):
    """Stockout risk prediction response"""
    predictions: List[StockoutRiskPrediction] = Field(..., description="List of predictions")
    model_version: str = Field(..., description="Model version used")
    processing_time_seconds: float = Field(..., description="Processing time")


class ProductResult(BaseModel):
    """Complete product analysis result"""
    sku: str = Field(..., description="Product SKU")
    product_name: str = Field(..., description="Product name")
    viability_score: float = Field(..., ge=0, le=1, description="Viability score")
    viability_class: str = Field(..., description="Viability class")
    recommended_price: float = Field(..., description="Recommended price")
    current_price: float = Field(..., description="Current price")
    margin_percent: float = Field(..., description="Margin percentage")
    stockout_risk_score: float = Field(..., ge=0, le=1, description="Stockout risk score")
    stockout_risk_level: RiskLevel = Field(..., description="Stockout risk level")
    cluster_id: Optional[int] = Field(None, description="Product cluster ID")
    rank: int = Field(..., description="Product rank based on viability")


class ResultsResponse(BaseModel):
    """Complete results response"""
    results: List[ProductResult] = Field(..., description="List of product results")
    total_products: int = Field(..., description="Total number of products")
    processing_timestamp: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
    model_versions: Dict[str, str] = Field(..., description="Model versions used")


class UploadResponse(BaseModel):
    """File upload response"""
    file_id: str = Field(..., description="Unique file ID")
    filename: str = Field(..., description="Original filename")
    file_size_bytes: int = Field(..., description="File size in bytes")
    total_rows: int = Field(..., description="Total rows in file")
    upload_timestamp: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    message: str = Field(..., description="Upload status message")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error detail")
    error_code: Optional[str] = Field(None, description="Error code")

