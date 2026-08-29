"""
resume_yolo_exp_v2.py - Resume YOLOv8n Exp v2 Training & Automated Test Evaluation
==================================================================================
Project: Cargo Vision Logistics System
Purpose: Resumes the YOLOv8n Exp v2 training from the exact saved checkpoint
         (runs/detect/runs/detect/cargo_yolo_exp_v2/weights/last.pt) without restarting
         from epoch 1, and upon completion of epoch 50 executes the full held-out test evaluation.
"""

import os
import sys

# Unbuffered stdout streaming
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import shutil
import json
import torch

try:
    cpu_cores = os.cpu_count() or 4
    threads = min(8, cpu_cores)
    torch.set_num_threads(threads)
    print(f"[SYSTEM] Configured PyTorch CPU threads: {threads} (Total CPU cores: {cpu_cores})", flush=True)
except Exception:
    pass

from ultralytics import YOLO

LAST_CHECKPOINT = os.path.join(PROJECT_ROOT, "runs", "detect", "runs", "detect", "cargo_yolo_exp_v2", "weights", "last.pt")
OUTPUT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cargo_yolo_exp_v2_best.pt")
DATA_YAML_PATH = os.path.join(PROJECT_ROOT, "deployment_dataset_expanded", "data.yaml")


def resume_training(checkpoint_path: str = LAST_CHECKPOINT, data_yaml: str = DATA_YAML_PATH, output_model: str = OUTPUT_MODEL_PATH):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found at: {checkpoint_path}")

    print("=" * 80, flush=True)
    print("CARGO VISION - RESUMING YOLOv8n EXP V2 TRAINING (EPOCH 14 -> 50)", flush=True)
    print("=" * 80, flush=True)
    print(f"Resume Checkpoint:   {checkpoint_path}", flush=True)
    print(f"Dataset YAML:        {data_yaml}", flush=True)
    print(f"Target Checkpoint:   {output_model}", flush=True)
    print("=" * 80, flush=True)

    # 1. Resume training
    model = YOLO(checkpoint_path)
    train_results = model.train(resume=True)

    # 2. Save best checkpoint
    save_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    best_weights_path = os.path.join(save_dir, "weights", "best.pt")
    if os.path.exists(best_weights_path):
        shutil.copy2(best_weights_path, output_model)
        print(f"\n[OK] Copied final best weights to: {output_model}", flush=True)
    elif os.path.exists(checkpoint_path):
        shutil.copy2(checkpoint_path, output_model)
        print(f"\n[OK] Copied final last weights to: {output_model}", flush=True)

    # 3. Evaluate on HELD-OUT TEST split only
    print("\n" + "=" * 80, flush=True)
    print("EVALUATING FINAL EXP V2 MODEL ON HELD-OUT TEST SPLIT...", flush=True)
    print("=" * 80, flush=True)
    best_model = YOLO(output_model)
    test_metrics = best_model.val(
        data=data_yaml,
        split="test",
        imgsz=640,
        batch=16,
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
        "model_architecture": "yolov8n.pt",
        "dataset": data_yaml,
        "classes_count": len(names_list),
        "epochs_trained": 50,
        "image_size": 640,
        "optimizer": "AdamW",
        "run_directory": save_dir,
        "output_model_path": output_model,
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
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Final evaluation report saved to: {report_path}", flush=True)
    print(f"Test Precision: {p * 100:.2f}% | Test Recall: {r * 100:.2f}% | Test mAP@50: {map50 * 100:.2f}% | Test mAP@50-95: {map50_95 * 100:.2f}%", flush=True)
    return summary


if __name__ == "__main__":
    resume_training()
