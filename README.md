# DropSmart — Product & Price Intelligence for Dropshipping Sellers

An end-to-end decision intelligence platform that enables dropshipping sellers to identify high-viability products, price them optimally, and predict stockout risks.

## Features

- **Product Viability Scoring**: ML-based prediction of sale probability within 30 days
- **Price Optimization**: Profit-maximizing price recommendations with MAP constraints
- **Stockout Risk Prediction**: Early warning system for inventory and lead-time risks
- **Product Clustering**: Group similar products for analog-based insights
- **Interactive Dashboard**: Streamlit UI for easy file upload and results visualization
- **RESTful API**: FastAPI backend for programmatic access

## Project Structure

```
drop-smart/
├── backend/          # FastAPI backend
├── frontend/         # Streamlit frontend
├── ml/              # ML models and pipelines
├── data/            # Data storage
├── tests/           # Test suite
├── scripts/         # Utility scripts
└── config/          # Configuration files
```

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed structure.

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd drop-smart
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running with Docker

```bash
docker-compose up --build
```

The application will be available at:
- Streamlit UI: http://localhost:8501
- FastAPI API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Running Locally

1. **Start FastAPI backend**
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```

2. **Start Streamlit frontend** (in a new terminal)
   ```bash
   cd frontend
   streamlit run main.py --server.port 8501
   ```

## Usage

1. Upload an Excel file with supplier data
2. System validates schema and processes data
3. View ranked products with viability scores
4. Check recommended prices and stockout risks
5. Export results as CSV

## API Endpoints

- `POST /upload` - Upload Excel file
- `POST /validate` - Validate Excel schema
- `POST /predict_viability` - Get viability predictions
- `POST /optimize_price` - Get optimized prices
- `POST /stockout_risk` - Get stockout risk predictions
- `GET /get_results` - Retrieve complete results

See [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) for detailed API documentation.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
```

## License

[Add your license here]

## Authors

Pransav M. Patel, Bilal Qader

