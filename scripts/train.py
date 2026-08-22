"""
YOLOv8 Model Training and Fine-Tuning Utility Script.

Usage:
    python scripts/train.py --epochs 100 --imgsz 640 --batch 16
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO
from src.rootinly.config import settings
from src.rootinly.logger import logger

def train_model(
    data_yaml: Path,
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    base_model: str = "yolov8n-seg.pt",
    output_dir: Path = None,
):
    output_dir = output_dir or (ROOT_DIR / "runs" / "segment")
    logger.info(f"Starting YOLOv8 Segmentation training with dataset '{data_yaml}'...")
    
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data configuration file not found: {data_yaml}")

    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=str(output_dir),
        name="crown_segmentation",
        save=True,
    )
    logger.info("Training complete!")
    return results

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Crown Segmentation Model")
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT_DIR / "data" / "data.yaml"),
        help="Path to data.yaml",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image input size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--base-model", type=str, default="yolov8n-seg.pt", help="Base model checkpoint")

    args = parser.parse_args()
    train_model(
        data_yaml=Path(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        base_model=args.base_model,
    )

if __name__ == "__main__":
    main()
