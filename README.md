# Rootinly AI - Unified Scalp & Hair Analysis Platform

A modular, production-ready Computer Vision and Machine Learning platform that unifies two clinical scalp analysis engines into a single FastAPI service and web interface:

1. **Crown View Hair Comparison**: Calculates exact follicular hair density and scalp visibility changes across clinical visits using Custom YOLOv8 Instance Segmentation and HSV colorimetry.
2. **Hairfall Stage Determiner**: Classifies clinical hairfall progression stage (Norwood Scale Stages 1–7) and model confidence score from scalp photographs using the local YOLOv8 Norwood Classification model (`models/best_norwood.pt`).

Both modules are served simultaneously from a single server entrypoint: `python main.py`.

---

## Architecture Overview

```
rootinly/
├── configs/                  # Centralized configuration & environment settings
│   ├── __init__.py
│   └── settings.py           # Typed settings dataclass with .env support
│
├── data/                     # Dataset (Train, Test, Valid annotations)
│   ├── train/
│   ├── test/
│   └── data.yaml
│
├── models/                   # Model weight checkpoints
│   ├── best.pt               # Custom trained YOLOv8 Segmentation weights (Module 1)
│   └── best_norwood.pt       # Custom trained YOLOv8 Norwood Classification weights (Module 2)
│
├── src/                      # Production source package
│   └── rootinly/
│       ├── __init__.py
│       ├── config.py         # Configuration re-exporter
│       ├── logger.py         # Logging setup & request ExecutionLogger
│       ├── schemas/          # Pydantic DTOs and API contract definitions
│       │   ├── __init__.py
│       │   ├── analysis.py   # VisitMetrics, Deltas, Visualizations, LogEntry
│       │   ├── response.py   # ComparisonResponse, HealthResponse, ErrorResponse
│       │   └── stage.py      # StageResponse, StageLogsListResponse
│       ├── core/             # AI & Computer Vision pipelines
│       │   ├── __init__.py
│       │   ├── preprocessor.py    # Image decoding, alignment, Base64 conversion
│       │   ├── segmentor.py       # YOLOv8 crown/hairline segmentation
│       │   ├── analyzer.py        # HSV scalp filtering & density calculation
│       │   ├── pipeline.py        # Crown comparison orchestration
│       │   └── stage_determiner.py# Local YOLOv8 Norwood classification service & logging
│       └── api/              # FastAPI Application layer
│           ├── __init__.py
│           ├── app.py        # Unified FastAPI factory & lifespan setup
│           ├── dependencies.py # Singleton dependency providers
│           └── routes/       # Modular API route controllers
│               ├── __init__.py
│               ├── health.py # /api/v1/health & system readiness
│               ├── comparison.py # /compare-crowns, /api/v1/crown/health & web UI
│               └── stage.py  # /predict-stage & /api/v1/stage/health
│
├── stage_determiner/         # Stage determiner storage
│   └── logs/                 # Dedicated stage prediction execution logs
│
├── static/                   # Web frontend assets
│   └── index.html            # Unified single-page clinical dashboard
│
├── scripts/                  # Command-line utilities
│   ├── run_server.py         # CLI launcher proxy
│   └── train.py              # YOLOv8 training / fine-tuning script
│
├── tests/                    # Comprehensive unit & integration tests
│   ├── __init__.py
│   ├── test_preprocessor.py  # Image processing unit tests
│   ├── test_analyzer.py      # Density calculation unit tests
│   ├── test_stage_determiner.py # Stage classification tests
│   └── test_api.py           # Unified FastAPI integration tests
│
├── main.py                   # Single production server entrypoint
├── Dockerfile                # Multi-stage production container
├── docker-compose.yml        # Container orchestration
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development & testing dependencies
├── pyproject.toml            # Package metadata & build definition
└── README.md                 # Documentation
```

---

## Features

- **Dual-Module Unified Service**: Both Crown Comparison and Stage Determination run from a single FastAPI server via `python main.py`.
- **Custom YOLOv8 Segmentation**: Organic hairline and crown ROI extraction eliminating background noise.
- **Follicular Density Analysis**: Precise pixel-level ratio calculations between hair follicles and exposed scalp.
- **Clinical Delta Tracking**: Automatic percentage change calculation across patient visits.
- **Follicular Visual Overlays**: High-resolution segmentation heatmaps (Green for hair, Orange for exposed scalp, Cyan for head contour).
- **Hairfall Stage Classification**: Scalp image evaluation delivering exact stage level and classification confidence percentage.
- **Clean Single-Page Frontend**: Modern, responsive dark-theme interface with zero external dependencies and seamless tab switching.
- **Dedicated Request Logging**: Timestamped diagnostic logs maintained for both Crown Comparison (`logs/`) and Stage Determiner (`stage_determiner/logs/`).
- **Production-Ready**: High-throughput async endpoints, comprehensive test suite, Docker containerization, and automated Swagger UI (`/docs`).

---

## Quick Start

### 1. Installation

```bash
# Clone repository and navigate to folder
cd rootinly

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Server

Run the unified server entrypoint:

```bash
python main.py
```

Optional CLI parameters:

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

#### 1. Unified System Health Check
`GET /api/v1/health`

**Response:**
```json
{
  "status": "healthy",
  "app_name": "Rootinly AI - Hair & Scalp Analysis Platform",
  "version": "2.0.0",
  "model_loaded": true,
  "model_path": "models/best.pt",
  "stage_model_loaded": true,
  "stage_model_path": "models/best_norwood.pt",
  "active_modules": ["crown_comparison", "stage_determiner"],
  "timestamp": "2026-08-26T19:46:55.123456"
}
```

#### 2. Module 1: Crown Hair Comparison
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
  "log_file": "logs/2026-08-26_19-46-55.log",
  "timestamp": "2026-08-26T19:46:55.123456",
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

#### 3. Module 2: Hairfall Stage Determiner
`POST /predict-stage`

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (file): Scalp photograph for classification

**Response Structure:**
```json
{
  "status": "success",
  "stage": "Stage 2",
  "confidence": 94.2,
  "duration_ms": 138.4,
  "duration_formatted": "138.4 ms",
  "log_file": "logs/stage_determiner/stage_predict_2026-08-26_19-46-55.log",
  "logs": [ ... ]
}
```

#### 4. Module Health Checks
- `GET /api/v1/crown/health`: Inspect YOLO crown segmentation module status and active model path.
- `GET /api/v1/stage/health`: Inspect YOLO Norwood stage classification module status and active model path.
- `GET /api/v1/health`: Overall system health and readiness across all modules.

---

## Logging Architecture

- **Growth Comparison Logs**: Written to [`logs/growth_comparison/`](logs/growth_comparison/) (e.g., `logs/growth_comparison/2026-08-26_19-46-55.log`).
- **Stage Determiner Logs**: Written to [`logs/stage_determiner/`](logs/stage_determiner/) (e.g., `logs/stage_determiner/stage_predict_2026-08-26_19-46-55.log`).

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
