"""FastAPI route handlers"""

import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, Response
import pandas as pd
import io

from backend.app.core.config import settings
from backend.app.services.pipeline_service import pipeline_service
from backend.app.models.schemas import (
    ValidationRequest,
    ValidationResponse,
    ValidationError,
    PredictViabilityRequest,
    ViabilityResponse,
    ViabilityPrediction,
    OptimizePriceRequest,
    PriceOptimizationResponse,
    PriceOptimization,
    StockoutRiskRequest,
    StockoutRiskResponse,
    StockoutRiskPrediction,
    ProductResult,
    ResultsResponse,
    UploadResponse,
    ErrorResponse,
    AvailabilityStatus,
    RiskLevel,
)

router = APIRouter()

# In-memory storage for demo (replace with database in production)
file_storage: Dict[str, Dict[str, Any]] = {}
results_storage: Dict[str, Dict[str, Any]] = {}


@router.post("/upload", response_model=UploadResponse, tags=["upload"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload Excel file with product data
    
    Accepts Excel files (.xlsx, .xls) and stores them for processing.
    Parses to DataFrame and stores in memory/temp storage.
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(settings.allowed_extensions)}"
        )
    
    # Validate file size
    contents = await file.read()
    if len(contents) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
        )
    
    # Generate file ID
    file_id = str(uuid.uuid4())
    
    # Save file
    file_path = settings.raw_data_dir / f"{file_id}{file_ext}"
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Parse to DataFrame
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        total_rows = len(df)
        
        # Store DataFrame in memory (for demo - use database in production)
        file_storage[file_id] = {
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size_bytes": len(contents),
            "total_rows": total_rows,
            "dataframe": df,  # Store DataFrame in memory
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read Excel file: {str(e)}"
        )
    
    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        file_size_bytes=len(contents),
        total_rows=total_rows,
        message="File uploaded successfully"
    )


@router.post("/validate", response_model=ValidationResponse, tags=["validation"])
async def validate_schema(request: ValidationRequest):
    """
    Validate Excel file schema
    
    Checks if uploaded file has all required fields and valid data types.
    Returns error messages if columns are missing.
    """
    if request.file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = file_storage[request.file_id]
    
    # Get DataFrame from storage
    if "dataframe" in file_info:
        df = file_info["dataframe"]
    else:
        # Fallback: read from file
        file_path = Path(file_info["file_path"])
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    # Required fields from schema
    required_fields = [
        "sku", "product_name", "cost", "price", "shipping_cost",
        "lead_time_days", "availability"
    ]
    
    optional_fields = [
        "description", "category", "weight_kg", "length_cm", "width_cm",
        "height_cm", "map_price", "duties", "supplier_name", "supplier_reliability_score"
    ]
    
    errors = []
    warnings = []
    missing_required = []
    missing_optional = []
    
    # Check required fields
    for field in required_fields:
        if field not in df.columns:
            missing_required.append(field)
            errors.append(ValidationError(
                field=field,
                message=f"Required field '{field}' is missing"
            ))
    
    # Check optional fields
    for field in optional_fields:
        if field not in df.columns:
            missing_optional.append(field)
            warnings.append(f"Optional field '{field}' is missing")
    
    # Validate data types for existing fields
    if "cost" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["cost"]):
            errors.append(ValidationError(
                field="cost",
                message="Field 'cost' must be numeric"
            ))
    
    if "price" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["price"]):
            errors.append(ValidationError(
                field="price",
                message="Field 'price' must be numeric"
            ))
    
    if "lead_time_days" in df.columns:
        if not pd.api.types.is_integer_dtype(df["lead_time_days"]):
            errors.append(ValidationError(
                field="lead_time_days",
                message="Field 'lead_time_days' must be integer"
            ))
    
    is_valid = len(errors) == 0
    
    return ValidationResponse(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        total_rows=len(df),
        total_columns=len(df.columns),
        missing_required_fields=missing_required,
        missing_optional_fields=missing_optional,
    )


@router.post("/predict_viability", response_model=ViabilityResponse, tags=["prediction"])
async def predict_viability(
    request: PredictViabilityRequest,
    top_k: Optional[int] = Query(None, description="Return top K products by viability score")
):
    """
    Predict product viability scores
    
    Returns probability of sale within 30 days for each product.
    Calls viability model and returns probabilities + top-K.
    """
    start_time = time.time()
    
    try:
        # Convert products to list of dicts
        products = [product.dict() for product in request.products]
        
        # Call pipeline service
        results = pipeline_service.predict_viability(products, top_k=top_k)
        
        # Format as ViabilityPrediction objects
        predictions = []
        for result in results:
            predictions.append(ViabilityPrediction(
                sku=result["sku"],
                viability_score=result["viability_score"],
                viability_class=result["viability_class"],
                shap_values=result.get("shap_values"),
            ))
        
        processing_time = time.time() - start_time
        
        return ViabilityResponse(
            predictions=predictions,
            model_version="1.0.0",
            processing_time_seconds=round(processing_time, 3),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict viability: {str(e)}")


@router.post("/optimize_price", response_model=PriceOptimizationResponse, tags=["optimization"])
async def optimize_price(request: OptimizePriceRequest):
    """
    Optimize product prices
    
    Returns recommended prices that maximize expected profit while respecting constraints.
    Calls price optimizer and returns recommended price.
    """
    start_time = time.time()
    
    try:
        # Convert products to list of dicts
        products = [product.dict() for product in request.products]
        
        # Call pipeline service
        results = pipeline_service.optimize_price(
            products,
            min_margin_percent=request.min_margin_percent,
            enforce_map=request.enforce_map
        )
        
        # Format as PriceOptimization objects
        optimizations = []
        for result in results:
            optimizations.append(PriceOptimization(
                sku=result["sku"],
                current_price=result["current_price"],
                recommended_price=result["recommended_price"],
                expected_profit=result["expected_profit"],
                current_profit=result["current_profit"],
                profit_improvement=result["profit_improvement"],
                margin_percent=result["margin_percent"],
                conversion_probability=result["conversion_probability"],
                map_constraint_applied=result["map_constraint_applied"],
                min_margin_constraint_applied=result["min_margin_constraint_applied"],
            ))
        
        processing_time = time.time() - start_time
        
        return PriceOptimizationResponse(
            optimizations=optimizations,
            model_version="1.0.0",
            processing_time_seconds=round(processing_time, 3),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize price: {str(e)}")


@router.post("/stockout_risk", response_model=StockoutRiskResponse, tags=["stockout"])
async def predict_stockout_risk(request: StockoutRiskRequest):
    """
    Predict stockout and lead-time risk
    
    Returns risk scores and factors for each product.
    Calls risk model.
    """
    start_time = time.time()
    
    try:
        # Convert products to list of dicts
        products = [product.dict() for product in request.products]
        
        # Call pipeline service
        results = pipeline_service.predict_stockout_risk(products)
        
        # Format as StockoutRiskPrediction objects
        predictions = []
        for result in results:
            # Convert risk level string to RiskLevel enum
            risk_level_str = result.get("risk_level", "low")
            try:
                risk_level = RiskLevel(risk_level_str.lower())
            except ValueError:
                risk_level = RiskLevel.LOW
            
            predictions.append(StockoutRiskPrediction(
                sku=result["sku"],
                risk_score=result["risk_score"],
                risk_level=risk_level,
                risk_factors=result.get("risk_factors", []),
                lead_time_risk=result.get("lead_time_risk", False),
                availability_risk=result.get("availability_risk", False),
            ))
        
        processing_time = time.time() - start_time
        
        return StockoutRiskResponse(
            predictions=predictions,
            model_version="1.0.0",
            processing_time_seconds=round(processing_time, 3),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict stockout risk: {str(e)}")


@router.get("/get_results", response_model=ResultsResponse, tags=["results"])
async def get_results(file_id: str):
    """
    Get complete analysis results for uploaded file
    
    Returns ranked list of products with all predictions and optimizations.
    Returns combined table with viability, price, risk, cluster.
    """
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Check if results are cached
    if file_id in results_storage:
        return results_storage[file_id]
    
    # Get DataFrame from storage
    file_info = file_storage[file_id]
    
    if "dataframe" in file_info:
        df = file_info["dataframe"].copy()
    else:
        # Fallback: read from file
        file_path = Path(file_info["file_path"])
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    try:
        # Process complete pipeline
        results_df = pipeline_service.process_complete_pipeline(df)
        
        # Convert to ProductResult objects
        results = []
        for idx, row in results_df.iterrows():
            # Convert risk level string to RiskLevel enum
            risk_level_str = str(row.get("stockout_risk_level", "low")).lower()
            try:
                risk_level = RiskLevel(risk_level_str)
            except ValueError:
                risk_level = RiskLevel.LOW
            
            results.append(ProductResult(
                sku=str(row.get("sku", f"SKU_{idx}")),
                product_name=str(row.get("product_name", "")),
                viability_score=float(row.get("viability_score", 0.0)),
                viability_class=str(row.get("viability_class", "low")),
                recommended_price=float(row.get("recommended_price", row.get("price", 0.0))),
                current_price=float(row.get("price", 0.0)),
                margin_percent=float(row.get("margin_percent", 0.0)),
                stockout_risk_score=float(row.get("stockout_risk_score", 0.0)),
                stockout_risk_level=risk_level,
                cluster_id=int(row.get("cluster_id")) if pd.notna(row.get("cluster_id")) else None,
                rank=int(row.get("rank", idx + 1)),
            ))
        
        # Create response
        response = ResultsResponse(
            results=results,
            total_products=len(results),
            model_versions={
                "viability": "1.0.0",
                "price_optimizer": "1.0.0",
                "stockout_risk": "1.0.0",
                "clustering": "1.0.0",
            }
        )
        
        # Cache results
        results_storage[file_id] = response
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process results: {str(e)}")


@router.get("/export_csv", tags=["export"])
async def export_csv(file_id: str):
    """
    Export ranked results as CSV file
    
    Returns a downloadable CSV file with all product analysis results.
    Ready for import into Amazon, Shopify, or ERP systems.
    """
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Get results (use cached if available, otherwise process)
    if file_id in results_storage:
        results = results_storage[file_id]
    else:
        # Process if not cached
        try:
            results = await get_results(file_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")
    
    # Convert ResultsResponse to dict if needed
    if hasattr(results, 'dict'):
        results_dict = results.dict()
    elif hasattr(results, 'model_dump'):
        results_dict = results.model_dump()
    else:
        results_dict = results
    
    # Prepare CSV data
    csv_data = []
    results_list = results_dict.get("results", [])
    
    for result in results_list:
        # Handle both dict and Pydantic model
        if hasattr(result, 'dict'):
            result_dict = result.dict()
        elif hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        else:
            result_dict = result
        
        # Handle RiskLevel enum
        risk_level = result_dict.get("stockout_risk_level", "low")
        if hasattr(risk_level, 'value'):
            risk_level = risk_level.value
        elif hasattr(risk_level, '__str__'):
            risk_level = str(risk_level)
        
        csv_data.append({
            "SKU": result_dict.get("sku", ""),
            "Product Name": result_dict.get("product_name", ""),
            "Rank": result_dict.get("rank", 0),
            "Viability Score": result_dict.get("viability_score", 0.0),
            "Viability Class": result_dict.get("viability_class", "low"),
            "Recommended Price": result_dict.get("recommended_price", 0.0),
            "Current Price": result_dict.get("current_price", 0.0),
            "Margin %": result_dict.get("margin_percent", 0.0),
            "Stockout Risk Score": result_dict.get("stockout_risk_score", 0.0),
            "Stockout Risk Level": risk_level,
            "Cluster ID": result_dict.get("cluster_id", "") if result_dict.get("cluster_id") is not None else "",
        })
    
    if not csv_data:
        raise HTTPException(status_code=404, detail="No results available for export")
    
    # Create DataFrame
    df = pd.DataFrame(csv_data)
    
    # Convert to CSV string
    csv_string = df.to_csv(index=False)
    csv_bytes = csv_string.encode('utf-8')
    
    # Get filename
    file_info = file_storage[file_id]
    filename = f"dropsmart_results_{file_id[:8]}.csv"
    
    # Return CSV file as response
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Type": "text/csv; charset=utf-8"
        }
    )
