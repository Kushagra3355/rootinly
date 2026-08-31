"""
Rootinly AI Unified Server Entrypoint (Dual-Module: Crown Comparison & Stage Determiner).

Usage:
    python main.py
    python main.py --port 5000 --host 0.0.0.0 --reload --no-browser
"""
import argparse
import sys
import threading
import webbrowser
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn
from src.rootinly.config import settings
from src.rootinly.api.app import app

__all__ = ["app", "main"]

def main():
    parser = argparse.ArgumentParser(
        description="Rootinly AI - Unified Hair & Scalp Analysis Server (Crown Comparison + Stage Determiner)"
    )
    parser.add_argument("--host", type=str, default=settings.HOST, help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=settings.PORT, help="Bind port (default: 5000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    parser.add_argument("--model-path", type=str, default=None, help="Custom YOLO segmentation model weights path")
    parser.add_argument("--stage-model-path", type=str, default=None, help="Custom YOLO Norwood stage model weights path")

    args = parser.parse_args()

    if args.model_path:
        settings.MODEL_PATH = Path(args.model_path)
    if args.stage_model_path:
        settings.STAGE_MODEL_PATH = Path(args.stage_model_path)

    server_url = f"http://localhost:{args.port}"

    if not args.no_browser and not args.reload:
        def open_browser():
            webbrowser.open(server_url)
        threading.Timer(1.5, open_browser).start()

    print(f"\n=======================================================")
    print(f"  {settings.APP_NAME}")
    print(f"  Version:        {settings.APP_VERSION}")
    print(f"  Web Interface:  {server_url}")
    print(f"  API Docs:       {server_url}/docs")
    print(f"  Health Check:   {server_url}/api/v1/health")
    print(f"  Active Modules:")
    print(f"    1. Crown Hair Comparison:       POST /compare-crowns")
    print(f"    2. Hairfall Stage Determiner:   POST /predict-stage")
    print(f"=======================================================\n")

    uvicorn.run(
        "src.rootinly.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

if __name__ == "__main__":
    main()

