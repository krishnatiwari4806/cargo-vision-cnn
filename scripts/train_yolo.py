"""
train_yolo.py - YOLO Object Detection Training Pipeline for 24 Cargo Classes
=============================================================================
Project: Cargo Vision Logistics System
Purpose: Trains a lightweight YOLO object detector (YOLOv8n) on the full 24-class
         deployment_dataset/ containing household, cargo, and vehicle-relevant classes.
         Saves the best checkpoint to models/cargo_yolo_24class_best.pt and
         records comprehensive evaluation metrics.
"""

import os
import sys
import shutil
import json
import argparse
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO


def train_yolo_detector(
    data_yaml: str = "deployment_dataset/data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 15,
    imgsz: int = 512,
    batch_size: int = 16,
    project_dir: str = "runs/detect",
    name: str = "cargo_yolo_24class",
    output_model_path: str = "models/cargo_yolo_24class_best.pt",
) -> Dict[str, Any]:
    """
    Trains YOLOv8 detector on deployment_dataset/ and exports best weights.
    """
    data_yaml_path = os.path.abspath(data_yaml)
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results/training", exist_ok=True)

    print("=" * 78)
    print("CARGO VISION - YOLO OBJECT DETECTION TRAINING PIPELINE (24 CLASSES)")
    print("=" * 78)
    print(f"Dataset YAML:        {data_yaml_path}")
    print(f"Base Pretrained:     {base_model}")
    print(f"Target Epochs:       {epochs}")
    print(f"Image Size:          {imgsz}x{imgsz}")
    print(f"Batch Size:          {batch_size}")
    print(f"Target Checkpoint:   {output_model_path}")
    print("=" * 78)

    # 1. Initialize YOLO model
    model = YOLO(base_model)

    # 2. Execute Training
    train_results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=project_dir,
        name=name,
        exist_ok=True,
        verbose=True,
        save=True,
        plots=True,
        workers=0,  # Single-process data loading for Windows stability
    )

    # 3. Locate best model checkpoint
    save_dir = str(getattr(train_results, "save_dir", os.path.join(project_dir, name)))
    best_weights_path = os.path.join(save_dir, "weights", "best.pt")
    last_weights_path = os.path.join(save_dir, "weights", "last.pt")

    if not os.path.exists(best_weights_path):
        import glob
        found = glob.glob(os.path.join(project_dir, "**", "best.pt"), recursive=True)
        if found:
            best_weights_path = found[0]

    if os.path.exists(best_weights_path):
        shutil.copy2(best_weights_path, output_model_path)
        print(f"\n[OK] Copied best weights to: {output_model_path}")
    elif os.path.exists(last_weights_path):
        shutil.copy2(last_weights_path, output_model_path)
        print(f"\n[OK] Copied last weights to: {output_model_path}")
    else:
        raise FileNotFoundError(f"Could not locate trained weights in: {save_dir}")

    # 4. Evaluate on held-out test split
    print("\n" + "=" * 78)
    print("EVALUATING MODEL ON HELD-OUT TEST SPLIT...")
    print("=" * 78)
    best_model = YOLO(output_model_path)
    val_metrics = best_model.val(
        data=data_yaml_path,
        split="test",
        imgsz=imgsz,
        batch=batch_size,
        verbose=True,
        workers=0,
    )

    # Extract metrics
    metrics_summary = {
        "model_architecture": base_model,
        "classes_count": 24,
        "epochs_trained": epochs,
        "image_size": imgsz,
        "best_model_path": output_model_path,
        "test_metrics": {
            "precision": round(float(val_metrics.results_dict.get("metrics/precision(B)", 0.0)), 4),
            "recall": round(float(val_metrics.results_dict.get("metrics/recall(B)", 0.0)), 4),
            "mAP50": round(float(val_metrics.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
            "mAP50_95": round(float(val_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
            "fitness": round(float(val_metrics.fitness), 4) if hasattr(val_metrics, "fitness") else None,
        },
    }

    summary_path = "results/training/yolo_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nSaved training summary to: {summary_path}")

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Cargo Vision 24-class dataset.")
    parser.add_argument("--data", type=str, default="deployment_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained YOLO weights (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size (default: 512)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--output", type=str, default="models/cargo_yolo_24class_best.pt", help="Output model path")
    args = parser.parse_args()

    train_yolo_detector(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        output_model_path=args.output,
    )


if __name__ == "__main__":
    main()
