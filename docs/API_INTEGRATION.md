# FastAPI Endpoints Integration with ML Modules

## ✅ Complete Integration

### Files Created/Updated

1. **`backend/app/services/pipeline_service.py`** - ML pipeline orchestration service
2. **`backend/app/api/routes.py`** - Updated with actual ML model calls

## Pipeline Service

The `MLPipelineService` class orchestrates all ML model calls:

### Features

- **Model Loading**: Automatically loads trained models from disk
- **Feature Preparation**: Prepares features from raw product data
- **Error Handling**: Graceful fallbacks when models aren't loaded
- **Batch Processing**: Efficient batch operations

### Methods

- `predict_viability()`: Calls viability model
- `optimize_price()`: Calls price optimizer
- `predict_stockout_risk()`: Calls stockout risk model
- `get_cluster_assignments()`: Gets cluster IDs
- `process_complete_pipeline()`: Runs all models and combines results

## API Endpoints

### 1. `POST /api/v1/upload`

**Functionality:**
- Accepts Excel file upload
- Parses to DataFrame using pandas
- Stores DataFrame in memory (for demo)
- Returns file_id for subsequent operations

**Implementation:**
```python
# Saves file to disk
# Parses to DataFrame
# Stores in file_storage with DataFrame
```

### 2. `POST /api/v1/validate`

**Functionality:**
- Validates Excel schema
- Checks for required fields
- Validates data types
- Returns error messages if columns are missing

**Implementation:**
- Uses stored DataFrame from upload
- Checks required fields: sku, product_name, cost, price, shipping_cost, lead_time_days, availability
- Validates data types
- Returns ValidationResponse with errors and warnings

### 3. `POST /api/v1/predict_viability`

**Functionality:**
- Calls viability model
- Returns probabilities + top-K products

**Implementation:**
```python
# Calls pipeline_service.predict_viability()
# Uses ViabilityModel to predict scores
# Returns top K products if top_k parameter provided
# Includes SHAP values if available
```

### 4. `POST /api/v1/optimize_price`

**Functionality:**
- Calls price optimizer
- Returns recommended price

**Implementation:**
```python
# Calls pipeline_service.optimize_price()
# Uses PriceOptimizer with ConversionModel
# Applies MAP and min-margin constraints
# Returns optimized prices with profit metrics
```

### 5. `POST /api/v1/stockout_risk`

**Functionality:**
- Calls risk model
- Returns risk scores and factors

**Implementation:**
```python
# Calls pipeline_service.predict_stockout_risk()
# Uses StockoutRiskModel
# Returns risk scores, levels, and factors
```

### 6. `GET /api/v1/get_results`

**Functionality:**
- Returns combined table with viability, price, risk, cluster
- Processes complete pipeline
- Returns ranked products

**Implementation:**
```python
# Calls pipeline_service.process_complete_pipeline()
# Runs all models:
#   - Viability predictions
#   - Price optimization
#   - Stockout risk
#   - Clustering
# Combines all results into ProductResult objects
# Ranks by viability score
```

## Model Loading

Models are loaded from:
- `data/models/viability/model.pkl`
- `data/models/price_optimizer/conversion_model.pkl`
- `data/models/stockout_risk/model.pkl`
- `data/models/clustering/model.pkl`

If models don't exist, endpoints return appropriate error messages.

## Error Handling

- **Model Not Loaded**: Returns HTTP 400 with clear error message
- **Invalid Data**: Validates input and returns errors
- **Processing Errors**: Catches exceptions and returns HTTP 500 with details
- **Graceful Fallbacks**: Some operations continue with defaults if models unavailable

## Usage Flow

1. **Upload File**: `POST /upload` → Returns file_id
2. **Validate**: `POST /validate` → Validates schema
3. **Get Results**: `GET /get_results?file_id=...` → Runs complete pipeline
   - Or call individual endpoints:
     - `POST /predict_viability`
     - `POST /optimize_price`
     - `POST /stockout_risk`

## Example Request/Response

### Upload File
```bash
POST /api/v1/upload
Content-Type: multipart/form-data

file: products.xlsx
```

Response:
```json
{
  "file_id": "uuid-here",
  "filename": "products.xlsx",
  "file_size_bytes": 12345,
  "total_rows": 100,
  "message": "File uploaded successfully"
}
```

### Get Results
```bash
GET /api/v1/get_results?file_id=uuid-here
```

Response:
```json
{
  "results": [
    {
      "sku": "SKU001",
      "product_name": "Product 1",
      "viability_score": 0.85,
      "viability_class": "high",
      "recommended_price": 49.99,
      "current_price": 45.00,
      "margin_percent": 30.0,
      "stockout_risk_score": 0.2,
      "stockout_risk_level": "low",
      "cluster_id": 5,
      "rank": 1
    },
    ...
  ],
  "total_products": 100,
  "model_versions": {...}
}
```

## Next Steps

1. Train models and save to expected paths
2. Test endpoints with real data
3. Add authentication/authorization
4. Add rate limiting
5. Replace in-memory storage with database
6. Add caching for results

