# Rootinly AI - Crown View Hair Comparison Engine

A modular, production-ready Computer Vision and Machine Learning microservice that calculates exact follicular hair density and scalp visibility changes across clinical visits using Custom YOLOv8 Instance Segmentation.

---

## Architecture Overview

```
rootinly_model/
├── configs/                  # Centralized configuration & environment settings
│   ├── __init__.py
│   └── settings.py           # Typed settings dataclass with .env support
│
├── data/                     # Dataset (Train, Test, Valid & Roboflow annotations)
│   ├── train/
│   ├── test/
│   └── data.yaml
│
├── models/                   # Model weight checkpoints
│   └── best.pt               # Custom trained YOLOv8 Segmentation weights
│
├── src/                      # Production source package
│   └── rootinly/
│       ├── __init__.py
│       ├── config.py         # Config re-exporter
│       ├── logger.py         # Structured logging & per-request ExecutionLogger
│       ├── schemas/          # Pydantic DTOs and API contract definitions
│       │   ├── __init__.py
│       │   ├── analysis.py   # Metrics, Deltas, Visualizations, LogEntry
│       │   └── response.py   # ComparisonResponse, HealthResponse, ErrorResponse
│       ├── core/             # AI & Computer Vision pipeline
│       │   ├── __init__.py
│       │   ├── preprocessor.py # Image decoding, alignment, Base64 conversion
│       │   ├── segmentor.py  # YOLOv8 crown/hairline segmentation
│       │   ├── analyzer.py   # HSV skin/scalp filtering, density & delta calculation
│       │   └── pipeline.py   # End-to-end orchestration
│       └── api/              # FastAPI Application layer
│           ├── __init__.py
│           ├── app.py        # Application factory & lifespan management
│           ├── dependencies.py # Singleton dependency injection
│           └── routes/       # Modular API route controllers
│               ├── __init__.py
│               ├── health.py # /api/v1/health & system status
│               └── comparison.py # /compare-crowns & frontend serving
│
├── static/                   # Web frontend assets
│   └── index.html            # Single-Page clinical dashboard UI
│
├── scripts/                  # Command-line utilities
│   ├── run_server.py         # CLI server launcher
│   └── train.py              # YOLOv8 training / fine-tuning script
│
├── tests/                    # Comprehensive unit & integration tests
│   ├── __init__.py
│   ├── test_preprocessor.py  # Image processing unit tests
│   ├── test_analyzer.py      # Density calculation unit tests
│   └── test_api.py           # FastAPI integration tests
│
├── main.py                   # Production root entrypoint
├── new.py                    # Backward compatibility proxy
├── Dockerfile                # Multi-stage production container
├── docker-compose.yml        # Container orchestration
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development & testing dependencies
├── pyproject.toml            # Package metadata & build definition
└── README.md                 # Documentation
```

---

## Features

- **Custom YOLOv8 Segmentation**: Organic hairline and crown ROI extraction eliminating background noise.
- **Follicular Density Analysis**: Precise pixel-level ratio calculations between hair follicles and scalp exposure.
- **Clinical Delta Calculations**: Automatic positive/negative delta tracking across visits.
- **Visual Overlays & Contours**: Color-coded segmentation heatmaps (Green for hair, Orange for exposed scalp, Cyan for head contour).
- **FastAPI Async Engine**: High-throughput REST API with automatic Swagger UI (`/docs`).
- **Clean Architecture & Separation of Concerns**: Decoupled domain models, core CV logic, schemas, and API routers.
- **Docker Ready**: Production containerization with health checks.

---

## Quick Start

### 1. Installation

```bash
# Clone repository and navigate to folder
cd rootinly_model

# Create and activate virtual environment (optional)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Server

Run the production entrypoint:

```bash
python main.py
```

Or with custom parameters:

```bash
python main.py --host 0.0.0.0 --port 5000 --reload --no-browser
```

Open your browser at `http://localhost:5000` to access the interactive web interface.

---

## API Documentation

FastAPI provides automated interactive API documentation:
- **Swagger UI**: `http://localhost:5000/docs`
- **ReDoc**: `http://localhost:5000/redoc`

### Key Endpoints

#### 1. System Health Check
`GET /api/v1/health`

**Response:**
```json
{
  "status": "healthy",
  "app_name": "Crown View Hair Comparison API",
  "version": "2.0.0",
  "model_loaded": true,
  "model_path": "models/best.pt",
  "timestamp": "2026-08-22T15:17:30.123456"
}
```

#### 2. Crown Comparison
`POST /compare-crowns`

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `last_visit_image` (file): Baseline visit photograph
  - `today_visit_image` (file): Follow-up visit photograph

**Response Structure:**
```json
{
  "status": "success",
  "pipeline": "yolov8_segmentation",
  "response_time_ms": 142.5,
  "response_time_formatted": "142.5 ms",
  "log_file": "logs/2026-08-22_15-17-30.log",
  "timestamp": "2026-08-22T15:17:30.123456",
  "previous_visit": {
    "hair_density_percent": 68.45,
    "scalp_visibility_percent": 31.55,
    "roi_total_pixels": 245000,
    "hair_pixels": 167702,
    "scalp_pixels": 77298
  },
  "today_visit": {
    "hair_density_percent": 74.20,
    "scalp_visibility_percent": 25.80,
    "roi_total_pixels": 245000,
    "hair_pixels": 181790,
    "scalp_pixels": 63210
  },
  "percentage_change": {
    "hair_density_change_pct": 8.40,
    "scalp_visibility_change_pct": -18.22
  },
  "visualizations": {
    "previous_visit_overlay_b64": "data:image/png;base64,...",
    "today_visit_overlay_b64": "data:image/png;base64,..."
  },
  "logs": [ ... ]
}
```

---

## Running Tests

Execute the automated test suite covering unit tests and API integration tests:

```bash
python -m unittest discover tests
```

---

## Training / Fine-Tuning YOLOv8

To re-train or fine-tune the crown segmentation model with new dataset annotations:

```bash
python scripts/train.py --data data/data.yaml --epochs 100 --batch 16 --imgsz 640
```

---

## Docker Deployment

### Build and run with Docker:

```bash
docker build -t rootinly-ai .
docker run -p 5000:5000 rootinly-ai
```

### Or using Docker Compose:

```bash
docker compose up --build
```
