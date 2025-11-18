# FastAPI Backend Implementation

## ✅ Completed Implementation

### File Structure

```
backend/app/
├── __init__.py
├── main.py                    # FastAPI application entry point
├── core/
│   ├── __init__.py
│   └── config.py             # Application configuration
├── models/
│   ├── __init__.py
│   └── schemas.py            # Pydantic request/response models
└── api/
    ├── __init__.py
    └── routes.py             # All API route handlers
```

## API Endpoints

All endpoints are prefixed with `/api/v1`:

### 1. `POST /api/v1/upload`
- **Purpose**: Upload Excel file with product data
- **Request**: Multipart form data with file
- **Response**: `UploadResponse` with file_id, filename, file size, row count
- **Features**:
  - Validates file extension (.xlsx, .xls)
  - Validates file size (max 50MB)
  - Stores file in `data/raw/` directory
  - Returns unique file_id for subsequent operations

### 2. `POST /api/v1/validate`
- **Purpose**: Validate Excel file schema
- **Request**: `ValidationRequest` with file_id
- **Response**: `ValidationResponse` with validation status, errors, warnings
- **Features**:
  - Checks for required fields (sku, product_name, cost, price, shipping_cost, lead_time_days, availability)
  - Checks for optional fields
  - Validates data types
  - Returns detailed error messages

### 3. `POST /api/v1/predict_viability`
- **Purpose**: Predict product viability scores
- **Request**: `PredictViabilityRequest` with list of products
- **Response**: `ViabilityResponse` with predictions for each product
- **Features**:
  - Returns viability score (0-1 probability)
  - Returns viability class (high/medium/low)
  - Includes SHAP values for explainability
  - Currently uses heuristic stub (ready for ML model integration)

### 4. `POST /api/v1/optimize_price`
- **Purpose**: Optimize product prices
- **Request**: `OptimizePriceRequest` with products and constraints
- **Response**: `PriceOptimizationResponse` with optimized prices
- **Features**:
  - Maximizes expected profit
  - Enforces MAP (Minimum Advertised Price) constraints
  - Enforces minimum margin percentage
  - Returns profit improvement metrics
  - Currently uses optimization stub (ready for ML model integration)

### 5. `POST /api/v1/stockout_risk`
- **Purpose**: Predict stockout and lead-time risk
- **Request**: `StockoutRiskRequest` with list of products
- **Response**: `StockoutRiskResponse` with risk predictions
- **Features**:
  - Returns risk score (0-1 probability)
  - Returns risk level (low/medium/high)
  - Identifies risk factors (lead time, availability)
  - Currently uses heuristic stub (ready for ML model integration)

### 6. `GET /api/v1/get_results`
- **Purpose**: Get complete analysis results for uploaded file
- **Request**: Query parameter `file_id`
- **Response**: `ResultsResponse` with ranked products
- **Features**:
  - Returns all predictions (viability, price, stockout risk)
  - Products ranked by viability score
  - Includes margin calculations
  - Results are cached for performance

## Pydantic Schemas

### Request Models
- `ProductInput`: Single product data
- `BulkProductInput`: Multiple products
- `ValidationRequest`: File validation request
- `PredictViabilityRequest`: Viability prediction request
- `OptimizePriceRequest`: Price optimization request
- `StockoutRiskRequest`: Stockout risk request

### Response Models
- `UploadResponse`: File upload confirmation
- `ValidationResponse`: Schema validation results
- `ViabilityResponse`: Viability predictions
- `PriceOptimizationResponse`: Price optimizations
- `StockoutRiskResponse`: Stockout risk predictions
- `ResultsResponse`: Complete analysis results
- `ErrorResponse`: Error information

### Enums
- `AvailabilityStatus`: in_stock, low_stock, out_of_stock, pre_order
- `RiskLevel`: low, medium, high

## Configuration

### Settings (`backend/app/core/config.py`)
- CORS configured for Streamlit frontend (localhost:8501)
- File upload limits (50MB max)
- Data directory paths
- API prefix configuration
- All settings can be overridden via environment variables

## Features

### ✅ Complete Implementation
- All 6 endpoints fully implemented
- Complete Pydantic schemas with validation
- CORS configured for Streamlit
- File upload and storage
- Error handling
- Type safety with Pydantic
- API documentation (auto-generated at `/docs`)

### 🔄 Ready for ML Integration
- Stub functions for all ML predictions
- Easy to replace with actual ML models
- Consistent interface for model integration
- Results caching for performance

## Running the API

### Development
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker-compose up backend
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

All endpoints are ready for testing. Example using curl:

```bash
# Health check
curl http://localhost:8000/health

# Upload file
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@products.xlsx"

# Validate
curl -X POST "http://localhost:8000/api/v1/validate" \
  -H "Content-Type: application/json" \
  -d '{"file_id": "your-file-id"}'
```

## Next Steps

1. Replace stub functions with actual ML models
2. Add database for persistent storage
3. Add authentication/authorization
4. Add rate limiting
5. Add request logging
6. Add unit and integration tests

