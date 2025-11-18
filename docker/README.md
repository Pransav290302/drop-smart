# Docker Configuration for DropSmart

## Overview

This directory contains Docker configuration files for running DropSmart in containers:
- **Backend**: FastAPI service on port 8000
- **Frontend**: Streamlit UI on port 8501

Both containers share the same codebase and communicate via an internal Docker network.

## Files

- `Dockerfile.backend` - FastAPI backend container
- `Dockerfile.frontend` - Streamlit frontend container
- `docker-compose.yml` - Orchestration for both services

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+

## Quick Start

### Build and Run

From the project root directory:

```bash
# Build and start both services
docker-compose -f docker/docker-compose.yml up --build

# Run in detached mode
docker-compose -f docker/docker-compose.yml up -d --build

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down
```

### Access Services

- **FastAPI Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Streamlit Frontend**: http://localhost:8501

## Architecture

```
┌─────────────────┐
│   Streamlit     │
│   Frontend      │
│   Port: 8501    │
└────────┬────────┘
         │
         │ HTTP
         │
┌────────▼────────┐
│   FastAPI       │
│   Backend       │
│   Port: 8000    │
└─────────────────┘
```

Both containers:
- Share the same codebase (mounted from parent directory)
- Use internal Docker network (`dropsmart-network`)
- Have access to shared volumes (`data/`, `config/`)
- Are CPU-only (no GPU dependencies)

## Environment Variables

### Backend
- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `DEBUG`: Debug mode (default: false)
- `PYTHONPATH`: Python path (default: /app)

### Frontend
- `API_BASE_URL`: Backend API URL (default: http://backend:8000)
- `STREAMLIT_SERVER_PORT`: Streamlit port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Streamlit address (default: 0.0.0.0)
- `PYTHONPATH`: Python path (default: /app)

## Volumes

- `../data:/app/data` - Data storage (models, uploads, outputs)
- `../config:/app/config` - Configuration files

## Health Checks

Both services include health checks:
- **Backend**: `GET /health` endpoint
- **Frontend**: Streamlit health endpoint

## Building Individual Services

### Backend Only
```bash
docker build -f docker/Dockerfile.backend -t dropsmart-backend ..
docker run -p 8000:8000 dropsmart-backend
```

### Frontend Only
```bash
docker build -f docker/Dockerfile.frontend -t dropsmart-frontend ..
docker run -p 8501:8501 -e API_BASE_URL=http://localhost:8000 dropsmart-frontend
```

## Troubleshooting

### Port Already in Use
```bash
# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Backend
  - "8502:8501"  # Frontend
```

### Container Won't Start
```bash
# Check logs
docker-compose -f docker/docker-compose.yml logs backend
docker-compose -f docker/docker-compose.yml logs frontend

# Rebuild without cache
docker-compose -f docker/docker-compose.yml build --no-cache
```

### Network Issues
```bash
# Verify network exists
docker network ls | grep dropsmart

# Inspect network
docker network inspect dropsmart-network
```

## Production Considerations

For production deployment:
1. Use environment-specific configuration files
2. Set up proper secrets management
3. Configure resource limits in docker-compose.yml
4. Use reverse proxy (nginx/traefik) for SSL termination
5. Set up logging aggregation
6. Configure backup strategy for data volumes

