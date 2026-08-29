"""
train_yolo_improved.py - Enhanced YOLO Object Detection Training Pipeline
==========================================================================
Project: Cargo Vision Logistics System
Purpose: Implements an enhanced YOLO training pipeline for 24 cargo classes with:
         - Native 640x640 resolution
         - Enhanced data augmentations (mosaic, HSV jitter, scale, translation)
         - Cosine learning rate schedule with warmup
         - Optimizer parameterization (AdamW, weight decay)
         - Model capacity comparison support (yolov8n vs yolov8s)
         - Explicit output saving to models/cargo_yolo_improved_best.pt
         - Preserves baseline models and training scripts intact.
"""

import os
import sys
import shutil
import json
import argparse
import glob
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO


def train_improved_yolo(
    data_yaml: str = "deployment_dataset/data.yaml",
    base_model: str = "yolov8s.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch_size: int = 8,
    lr0: float = 0.001,
    lrf: float = 0.01,
    weight_decay: float = 0.0005,
    warmup_epochs: int = 3,
    project_dir: str = "runs/detect",
    name: str = "cargo_yolo_improved",
    output_model_path: str = "models/cargo_yolo_improved_best.pt",
) -> Dict[str, Any]:
    """
    Trains an improved YOLO detector on deployment_dataset/ with fine-tuned hyperparameters.
    """
    data_yaml_path = os.path.abspath(data_yaml)
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results/training", exist_ok=True)

    print("=" * 78)
    print("CARGO VISION - ENHANCED YOLO TRAINING PIPELINE (24 CLASSES)")
    print("=" * 78)
    print(f"Dataset YAML:        {data_yaml_path}")
    print(f"Base Pretrained:     {base_model}")
    print(f"Target Epochs:       {epochs}")
    print(f"Resolution:          {imgsz}x{imgsz} px (Native)")
    print(f"Batch Size:          {batch_size}")
    print(f"Initial LR (lr0):    {lr0}")
    print(f"Final LR Frac (lrf): {lrf}")
    print(f"Weight Decay:        {weight_decay}")
    print(f"Warmup Epochs:       {warmup_epochs}")
    print(f"Target Checkpoint:   {output_model_path}")
    print("=" * 78)

    # 1. Initialize YOLO model
    model = YOLO(base_model)

    # 2. Execute Training with tailored augmentations & hyperparameters
    train_results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        lr0=lr0,
        lrf=lrf,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        optimizer="AdamW",
        cos_lr=True,
        # Augmentation hyperparameters
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=8,
        # Project configuration
        project=project_dir,
        name=name,
        exist_ok=True,
        verbose=True,
        save=True,
        plots=True,
        workers=0,  # Single worker for Windows CPU stability
    )

    # 3. Locate best model checkpoint
    save_dir = str(getattr(train_results, "save_dir", os.path.join(project_dir, name)))
    best_weights_path = os.path.join(save_dir, "weights", "best.pt")
    last_weights_path = os.path.join(save_dir, "weights", "last.pt")

    if not os.path.exists(best_weights_path):
        found = glob.glob(os.path.join(project_dir, "**", "best.pt"), recursive=True)
        if found:
            best_weights_path = found[-1]

    if os.path.exists(best_weights_path):
        shutil.copy2(best_weights_path, output_model_path)
        print(f"\n[OK] Copied improved best weights to: {output_model_path}")
    elif os.path.exists(last_weights_path):
        shutil.copy2(last_weights_path, output_model_path)
        print(f"\n[OK] Copied improved last weights to: {output_model_path}")
    else:
        raise FileNotFoundError(f"Could not locate trained weights in: {save_dir}")

    # 4. Evaluate on held-out test split
    print("\n" + "=" * 78)
    print("EVALUATING IMPROVED MODEL ON HELD-OUT TEST SPLIT...")
    print("=" * 78)
    best_model = YOLO(output_model_path)
    test_metrics = best_model.val(
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
        "optimizer": "AdamW",
        "learning_rate_initial": lr0,
        "learning_rate_final": lr0 * lrf,
        "improved_model_path": output_model_path,
        "test_metrics": {
            "precision": round(float(test_metrics.results_dict.get("metrics/precision(B)", 0.0)), 4),
            "recall": round(float(test_metrics.results_dict.get("metrics/recall(B)", 0.0)), 4),
            "mAP50": round(float(test_metrics.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
            "mAP50_95": round(float(test_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
            "fitness": round(float(test_metrics.fitness), 4) if hasattr(test_metrics, "fitness") else None,
        },
    }

    summary_path = "results/training/yolo_improved_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nSaved improved training summary to: {summary_path}")

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced YOLO on Cargo Vision 24-class dataset.")
    parser.add_argument("--data", type=str, default="deployment_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Pretrained YOLO model (default: yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate (default: 0.001)")
    parser.add_argument("--output", type=str, default="models/cargo_yolo_improved_best.pt", help="Output model path")
    args = parser.parse_args()

    train_improved_yolo(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        lr0=args.lr0,
        output_model_path=args.output,
    )


if __name__ == "__main__":
    main()
