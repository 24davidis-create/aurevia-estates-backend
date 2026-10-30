# Aurevia Estates — Backend (FastAPI + Machine Learning)

> *"Where Every Property Finds Its Worth."*

High-performance FastAPI service serving trained Machine Learning models (Random Forest, Decision Tree, Ridge Regression, and KNN) trained on 12,000 Indian metro residential listings across 18 features.

## Models Included
- **Random Forest Regressor** (`random_forest.joblib`): Top accuracy ($R^2 \approx 88.5\%$, MAE $\approx \pm ₹10.6$ Lakhs)
- **Decision Tree Regressor** (`decision_tree.joblib`): CART tree model
- **Linear Regression (Ridge)** (`linear_regression.joblib`): Lowest latency baseline
- **K-Nearest Neighbors** (`knn.joblib`): Spatial proximity & property similarity

## API Endpoints
- `GET /` — Service status & available endpoints
- `GET /api/health` — Backend and 4-model health status
- `GET /api/features` — Feature metadata, categorical options, and numeric ranges
- `GET /api/models/metrics` — Model benchmarking metrics, $R^2$, MAE, RMSE, confusion matrices
- `GET /api/dataset/summary` — Overview of the 12,000-listing dataset
- `POST /api/predict` — Submit 18 property features to receive real-time valuations

## Local Development

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Ensure `FRONTEND_ORIGINS` allows your local frontend:
```env
FRONTEND_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
PORT=8000
```

### 3. Run Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
API docs available at: `http://localhost:8000/docs`

## Render Deployment (Web Service)
- **Service Type**: Web Service
- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  - `FRONTEND_ORIGINS`: `https://<your-frontend-app-name>.onrender.com`
  - `PYTHON_VERSION`: `3.11.9`
