# DropSmart Project Structure

## Complete Folder Tree

```
drop-smart/
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README.md
├── requirements.txt
├── requirements-dev.txt
│
├── backend/                          # FastAPI Backend
│   ├── __init__.py
│   ├── main.py                      # FastAPI app entry point
│   ├── config.py                    # Configuration settings
│   │
│   ├── api/                         # API routes
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── upload.py            # POST /upload endpoint
│   │   │   ├── validate.py          # POST /validate endpoint
│   │   │   ├── predict.py           # POST /predict_viability endpoint
│   │   │   ├── optimize.py          # POST /optimize_price endpoint
│   │   │   ├── stockout.py          # POST /stockout_risk endpoint
│   │   │   └── results.py           # GET /get_results endpoint
│   │   │
│   │   └── schemas/                 # Pydantic models
│   │       ├── __init__.py
│   │       ├── upload.py            # Upload request/response schemas
│   │       ├── validation.py        # Validation schemas
│   │       ├── prediction.py        # Prediction schemas
│   │       └── results.py           # Results schemas
│   │
│   ├── services/                    # Business logic layer
│   │   ├── __init__.py
│   │   ├── file_service.py          # File upload/handling logic
│   │   ├── validation_service.py    # Schema validation logic
│   │   └── ml_service.py            # ML model orchestration
│   │
│   └── utils/                       # Backend utilities
│       ├── __init__.py
│       ├── exceptions.py            # Custom exceptions
│       └── logger.py               # Logging configuration
│
├── frontend/                        # Streamlit Frontend
│   ├── __init__.py
│   ├── main.py                     # Streamlit app entry point
│   ├── config.py                   # Frontend configuration
│   │
│   ├── pages/                      # Streamlit pages
│   │   ├── __init__.py
│   │   ├── 1_🏠_Home.py            # Home/Upload page
│   │   ├── 2_✅_Validation.py      # Validation page
│   │   ├── 3_📊_Dashboard.py       # Main dashboard
│   │   ├── 4_🔍_Product_Detail.py  # Product detail view
│   │   └── 5_📥_Export.py          # CSV export page
│   │
│   ├── components/                 # Reusable UI components
│   │   ├── __init__.py
│   │   ├── file_uploader.py        # File upload component
│   │   ├── results_table.py        # Results table component
│   │   ├── shap_visualization.py   # SHAP plots component
│   │   ├── metrics_display.py      # Metrics display component
│   │   └── export_button.py        # Export functionality
│   │
│   └── utils/                      # Frontend utilities
│       ├── __init__.py
│       ├── api_client.py           # FastAPI client wrapper
│       ├── formatters.py           # Data formatting utilities
│       └── session_state.py        # Session state management
│
├── ml/                              # ML Models & Pipelines
│   ├── __init__.py
│   ├── config.py                   # ML configuration
│   │
│   ├── data/                       # Data processing
│   │   ├── __init__.py
│   │   ├── ingestion.py            # Excel ingestion module
│   │   ├── validation.py           # Data validation
│   │   ├── normalization.py        # Data normalization
│   │   └── preprocessing.py        # Data preprocessing
│   │
│   ├── features/                   # Feature Engineering
│   │   ├── __init__.py
│   │   ├── engineering.py          # Main feature engineering
│   │   ├── cost_features.py       # Landed cost, margin calculations
│   │   ├── weight_features.py     # Volumetric weight, size tier
│   │   ├── time_features.py       # Lead-time, seasonality
│   │   └── embeddings.py          # Product embeddings (MiniLM)
│   │
│   ├── models/                     # ML Models
│   │   ├── __init__.py
│   │   ├── base_model.py          # Base model interface
│   │   ├── viability/             # Viability model
│   │   │   ├── __init__.py
│   │   │   ├── model.py          # LightGBM + Logistic Regression
│   │   │   ├── trainer.py        # Training script
│   │   │   └── explainer.py      # SHAP explainer
│   │   │
│   │   ├── price_optimizer/       # Price optimization
│   │   │   ├── __init__.py
│   │   │   ├── optimizer.py      # Price optimization logic
│   │   │   ├── conversion_model.py # Conversion probability model
│   │   │   └── constraints.py    # MAP and margin constraints
│   │   │
│   │   ├── stockout_risk/         # Stockout risk model
│   │   │   ├── __init__.py
│   │   │   ├── model.py          # Risk prediction model
│   │   │   └── trainer.py        # Training script
│   │   │
│   │   └── clustering/            # Product clustering
│   │       ├── __init__.py
│   │       ├── clusterer.py       # K-means/HDBSCAN clustering
│   │       └── embeddings.py     # SentenceTransformer embeddings
│   │
│   ├── pipeline/                   # ML Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── pipeline.py            # Main pipeline orchestrator
│   │   └── steps.py               # Pipeline steps
│   │
│   └── evaluation/                 # Model evaluation
│       ├── __init__.py
│       ├── metrics.py             # Evaluation metrics
│       ├── plots.py               # Calibration plots, SHAP plots
│       └── ablation.py            # Ablation study scripts
│
├── data/                            # Data storage
│   ├── raw/                        # Raw input files
│   ├── processed/                  # Processed data
│   ├── models/                     # Trained model artifacts
│   │   ├── viability/
│   │   ├── price_optimizer/
│   │   ├── stockout_risk/
│   │   └── clustering/
│   ├── outputs/                    # Generated outputs
│   └── .gitkeep                    # Keep folder in git
│
├── tests/                           # Tests
│   ├── __init__.py
│   ├── conftest.py                 # Pytest configuration
│   │
│   ├── unit/                       # Unit tests
│   │   ├── __init__.py
│   │   ├── test_ingestion.py
│   │   ├── test_feature_engineering.py
│   │   ├── test_viability_model.py
│   │   ├── test_price_optimizer.py
│   │   └── test_stockout_risk.py
│   │
│   ├── integration/                # Integration tests
│   │   ├── __init__.py
│   │   ├── test_api_endpoints.py
│   │   ├── test_ml_pipeline.py
│   │   └── test_streamlit_flow.py
│   │
│   └── fixtures/                   # Test fixtures
│       ├── sample_data.xlsx
│       └── mock_models/
│
├── scripts/                         # Utility scripts
│   ├── train_models.py             # Train all models
│   ├── evaluate_models.py          # Run evaluation
│   ├── generate_sample_data.py     # Generate test data
│   └── setup_data.py               # Setup data directories
│
├── config/                          # Configuration files
│   ├── app_config.yaml             # Application configuration
│   ├── model_config.yaml           # Model hyperparameters
│   └── schema_config.yaml          # Excel schema definitions
│
└── docs/                            # Documentation
    ├── DropSmart_PRD.md            # Product Requirements Document
    ├── PROJECT_STRUCTURE.md        # This file
    ├── API_DOCUMENTATION.md        # API documentation
    └── DEPLOYMENT.md               # Deployment guide
```

---

## Folder Explanations

### Root Level Files

- **`.dockerignore`**: Excludes files from Docker builds (similar to .gitignore)
- **`.gitignore`**: Git ignore patterns for Python, data files, models, etc.
- **`docker-compose.yml`**: Orchestrates FastAPI and Streamlit services
- **`Dockerfile`**: Multi-stage Docker build for the application
- **`README.md`**: Project overview, setup instructions, usage guide
- **`requirements.txt`**: Production Python dependencies
- **`requirements-dev.txt`**: Development dependencies (pytest, black, etc.)

---

### `backend/` - FastAPI Backend

**Purpose**: RESTful API backend that handles file uploads, validation, and ML model inference.

#### `backend/main.py`
- FastAPI application instance
- CORS configuration
- Router registration
- Application lifecycle management

#### `backend/config.py`
- Environment variables
- API settings (ports, timeouts)
- Path configurations

#### `backend/api/routes/`
- **`upload.py`**: `POST /upload` - Handles Excel file uploads
- **`validate.py`**: `POST /validate` - Validates Excel schema
- **`predict.py`**: `POST /predict_viability` - Returns viability predictions
- **`optimize.py`**: `POST /optimize_price` - Returns optimized prices
- **`stockout.py`**: `POST /stockout_risk` - Returns stockout risk predictions
- **`results.py`**: `GET /get_results` - Retrieves complete analysis results

#### `backend/api/schemas/`
- Pydantic models for request/response validation
- Type-safe data structures for API communication

#### `backend/services/`
- Business logic layer (separated from routes)
- **`file_service.py`**: File handling, storage, retrieval
- **`validation_service.py`**: Schema validation logic
- **`ml_service.py`**: Orchestrates ML pipeline calls

#### `backend/utils/`
- Shared utilities, custom exceptions, logging setup

---

### `frontend/` - Streamlit Frontend

**Purpose**: User interface for uploading files, viewing results, and exporting data.

#### `frontend/main.py`
- Streamlit app entry point
- Page routing configuration
- Global app settings

#### `frontend/pages/`
- **`1_🏠_Home.py`**: File upload interface, initial landing page
- **`2_✅_Validation.py`**: Displays validation results and errors
- **`3_📊_Dashboard.py`**: Main results table with ranked products
- **`4_🔍_Product_Detail.py`**: Individual product details with SHAP visualizations
- **`5_📥_Export.py`**: CSV export functionality

#### `frontend/components/`
- Reusable UI components:
  - File uploader widget
  - Results table with sorting/filtering
  - SHAP visualization charts
  - Metrics display cards
  - Export button with download

#### `frontend/utils/`
- API client wrapper for FastAPI calls
- Data formatting utilities
- Session state management helpers

---

### `ml/` - Machine Learning Module

**Purpose**: All ML models, feature engineering, and pipeline orchestration.

#### `ml/data/`
- **`ingestion.py`**: Excel file parsing into DataFrames
- **`validation.py`**: Data quality checks
- **`normalization.py`**: Currency, unit normalization
- **`preprocessing.py`**: Data cleaning and preparation

#### `ml/features/`
- **`engineering.py`**: Main feature engineering orchestrator
- **`cost_features.py`**: Landed cost, margin % calculations
- **`weight_features.py`**: Volumetric weight, size tier classification
- **`time_features.py`**: Lead-time buckets, seasonality indicators
- **`embeddings.py`**: Product title embeddings using SentenceTransformers

#### `ml/models/`
- **`base_model.py`**: Abstract base class for all models
- **`viability/`**: Viability prediction model (LightGBM + Logistic Regression)
- **`price_optimizer/`**: Price optimization with constraints
- **`stockout_risk/`**: Stockout/lead-time risk prediction
- **`clustering/`**: Product clustering (K-means/HDBSCAN)

#### `ml/pipeline/`
- **`pipeline.py`**: Main orchestration of all ML steps
- **`steps.py`**: Individual pipeline steps (ingestion → features → models → results)

#### `ml/evaluation/`
- **`metrics.py`**: ROC-AUC, PR-AUC, calibration metrics
- **`plots.py`**: SHAP plots, calibration curves
- **`ablation.py`**: Ablation study scripts

---

### `data/` - Data Storage

**Purpose**: Stores input files, processed data, trained models, and outputs.

- **`raw/`**: Original Excel files uploaded by users
- **`processed/`**: Cleaned and processed DataFrames
- **`models/`**: Trained model artifacts (pickle/joblib files)
- **`outputs/`**: Generated CSV exports and analysis results

---

### `tests/` - Testing

**Purpose**: Unit and integration tests for all components.

- **`unit/`**: Unit tests for individual functions/classes
- **`integration/`**: End-to-end tests for API endpoints and ML pipeline
- **`fixtures/`**: Sample data files and mock models for testing

---

### `scripts/` - Utility Scripts

**Purpose**: Standalone scripts for training, evaluation, and setup.

- **`train_models.py`**: Train all ML models
- **`evaluate_models.py`**: Run evaluation metrics and generate reports
- **`generate_sample_data.py`**: Create synthetic test data
- **`setup_data.py`**: Initialize data directory structure

---

### `config/` - Configuration Files

**Purpose**: YAML/JSON configuration files for easy parameter tuning.

- **`app_config.yaml`**: Application settings (ports, paths, etc.)
- **`model_config.yaml`**: Model hyperparameters and settings
- **`schema_config.yaml`**: Excel schema definitions and required fields

---

### `docs/` - Documentation

**Purpose**: Project documentation and guides.

- **`DropSmart_PRD.md`**: Product Requirements Document
- **`PROJECT_STRUCTURE.md`**: This structure document
- **`API_DOCUMENTATION.md`**: API endpoint documentation
- **`DEPLOYMENT.md`**: Docker deployment instructions

---

## Key Design Principles

1. **Separation of Concerns**: Backend, frontend, and ML are clearly separated
2. **Modularity**: Each module can be developed and tested independently
3. **Scalability**: Structure supports future additions (multi-vendor, authentication)
4. **Maintainability**: Clear organization makes code easy to navigate and update
5. **Docker-Ready**: Structure supports containerization with clear service boundaries

---

## Next Steps

1. Create the folder structure
2. Initialize Python packages with `__init__.py` files
3. Set up Docker configuration
4. Create base configuration files
5. Implement modules incrementally following this structure

