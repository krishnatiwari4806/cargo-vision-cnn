"""
train_yolo_exp_v2.py - Live-Streaming Controlled YOLOv8n Training Pipeline (Exp v2)
===================================================================================
Project: Cargo Vision Logistics System
Purpose: Trains YOLOv8n on deployment_dataset_expanded/ for 50 epochs with live terminal output:
         - Live stdout flushing (unbuffered)
         - Multi-threaded CPU optimization (torch.set_num_threads)
         - Native 640x640 resolution
         - AdamW optimizer (lr0=0.001, lrf=0.01, cos_lr=True, weight_decay=0.0005)
         - Warmup: 3.0 epochs
         - Augmentations: HSV, rotation (10 deg), translation (0.1), scale (0.2), mosaic=1.0, close_mosaic=8
         - Evaluates validation split after every epoch
         - Saves best checkpoint to models/cargo_yolo_exp_v2_best.pt (strictly preserving previous checkpoints)
         - Evaluates on held-out test split with per-class and confusion matrix reports.
"""

import os
import sys

# Force immediate unbuffered stdout streaming
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

import shutil
import json
import argparse
import glob
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
# Optimize CPU threading for multi-core processors
try:
    cpu_cores = os.cpu_count() or 4
    threads = min(8, cpu_cores)
    torch.set_num_threads(threads)
    print(f"[SYSTEM] Configured PyTorch CPU threads: {threads} (Total CPU cores: {cpu_cores})", flush=True)
except Exception as e:
    print(f"[SYSTEM] Thread config notice: {e}", flush=True)

from ultralytics import YOLO


def train_exp_v2_yolo(
    data_yaml: str = "deployment_dataset_expanded/data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    lr0: float = 0.001,
    lrf: float = 0.01,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    close_mosaic: int = 8,
    project_dir: str = "runs/detect",
    name: str = "cargo_yolo_exp_v2",
    output_model_path: str = "models/cargo_yolo_exp_v2_best.pt",
) -> Dict[str, Any]:
    """
    Executes controlled YOLOv8n training experiment v2 on deployment_dataset_expanded/.
    """
    data_yaml_path = os.path.abspath(data_yaml)
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "training"), exist_ok=True)

    print("=" * 80, flush=True)
    print("CARGO VISION - LIVE YOLOv8n TRAINING EXPERIMENT (EXP V2)", flush=True)
    print("=" * 80, flush=True)
    print(f"Dataset YAML:        {data_yaml_path}", flush=True)
    print(f"Model Architecture:  {base_model}", flush=True)
    print(f"Target Epochs:       {epochs}", flush=True)
    print(f"Image Resolution:    {imgsz}x{imgsz} px", flush=True)
    print(f"Batch Size:          {batch_size}", flush=True)
    print(f"Optimizer:           AdamW", flush=True)
    print(f"Initial LR (lr0):    {lr0}", flush=True)
    print(f"Final LR Ratio:      {lrf} (Cosine Annealing)", flush=True)
    print(f"Weight Decay:        {weight_decay}", flush=True)
    print(f"Warmup Epochs:       {warmup_epochs}", flush=True)
    print(f"Close Mosaic Epochs: {close_mosaic}", flush=True)
    print(f"Run Directory:       {os.path.join(project_dir, name)}", flush=True)
    print(f"Target Checkpoint:   {output_model_path}", flush=True)
    print("=" * 80, flush=True)

    # 1. Initialize YOLO model
    model = YOLO(base_model)

    # 2. Train model
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
        # Augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=close_mosaic,
        # Execution & Saving
        project=project_dir,
        name=name,
        exist_ok=True,
        verbose=True,
        val=True,
        save=True,
        plots=True,
        workers=0,  # Single-process worker for optimal Windows CPU throughput
    )

    # 3. Locate best weights
    save_dir = str(getattr(train_results, "save_dir", os.path.join(project_dir, name)))
    best_weights_path = os.path.join(save_dir, "weights", "best.pt")
    last_weights_path = os.path.join(save_dir, "weights", "last.pt")

    if not os.path.exists(best_weights_path):
        found = glob.glob(os.path.join(project_dir, name, "**", "best.pt"), recursive=True)
        if found:
            best_weights_path = found[-1]

    if os.path.exists(best_weights_path):
        shutil.copy2(best_weights_path, output_model_path)
        print(f"\n[OK] Copied best weights to: {output_model_path}", flush=True)
    elif os.path.exists(last_weights_path):
        shutil.copy2(last_weights_path, output_model_path)
        print(f"\n[OK] Copied last weights to: {output_model_path}", flush=True)
    else:
        raise FileNotFoundError(f"Could not locate trained weights in: {save_dir}")

    # 4. Evaluate on HELD-OUT TEST split only
    print("\n" + "=" * 80, flush=True)
    print("EVALUATING EXPANDED MODEL ON HELD-OUT TEST SPLIT...", flush=True)
    print("=" * 80, flush=True)
    best_model = YOLO(output_model_path)
    test_metrics = best_model.val(
        data=data_yaml_path,
        split="test",
        imgsz=imgsz,
        batch=batch_size,
        verbose=True,
        workers=0,
    )

    p = round(float(test_metrics.results_dict.get("metrics/precision(B)", 0.0)), 4)
    r = round(float(test_metrics.results_dict.get("metrics/recall(B)", 0.0)), 4)
    map50 = round(float(test_metrics.results_dict.get("metrics/mAP50(B)", 0.0)), 4)
    map50_95 = round(float(test_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4)

    # Per-class metrics
    per_class = {}
    if isinstance(best_model.names, dict):
        names_list = [best_model.names[i] for i in range(len(best_model.names))]
    else:
        names_list = list(best_model.names)

    maps_per_class = getattr(test_metrics.box, "maps", [])
    p_per_class = getattr(test_metrics.box, "p", [])
    r_per_class = getattr(test_metrics.box, "r", [])

    for i, cname in enumerate(names_list):
        c_map50 = round(float(maps_per_class[i]), 4) if len(maps_per_class) > i else 0.0
        c_p = round(float(p_per_class[i]), 4) if len(p_per_class) > i else 0.0
        c_r = round(float(r_per_class[i]), 4) if len(r_per_class) > i else 0.0
        per_class[cname] = {
            "precision": c_p,
            "recall": c_r,
            "mAP50": c_map50,
        }

    # Confusion matrix extraction
    confusion_matrix_data = None
    if hasattr(test_metrics, "confusion_matrix") and test_metrics.confusion_matrix is not None:
        try:
            cm = test_metrics.confusion_matrix.matrix
            confusion_matrix_data = cm.tolist() if hasattr(cm, "tolist") else str(cm)
        except Exception:
            pass

    summary = {
        "model_architecture": base_model,
        "dataset": data_yaml,
        "classes_count": 24,
        "epochs_trained": epochs,
        "image_size": imgsz,
        "optimizer": "AdamW",
        "learning_rate_initial": lr0,
        "learning_rate_final": lr0 * lrf,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "close_mosaic": close_mosaic,
        "run_directory": os.path.join(project_dir, name),
        "output_model_path": output_model_path,
        "test_metrics": {
            "precision": p,
            "recall": r,
            "mAP50": map50,
            "mAP50-95": map50_95,
        },
        "per_class_metrics": per_class,
        "confusion_matrix": confusion_matrix_data,
    }

    report_path = os.path.join(PROJECT_ROOT, "results", "training", "cargo_yolo_exp_v2_eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Evaluation report saved to: {report_path}", flush=True)
    print(f"Test Precision: {p * 100:.2f}% | Test Recall: {r * 100:.2f}% | Test mAP@50: {map50 * 100:.2f}% | Test mAP@50-95: {map50_95 * 100:.2f}%", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n Exp V2 on deployment_dataset_expanded.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    args = parser.parse_args()

    train_exp_v2_yolo(epochs=args.epochs, batch_size=args.batch, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
