"""
evaluate_yolo.py - Comprehensive Evaluation Suite for 24-Class YOLO Object Detector
===================================================================================
Project: Cargo Vision Logistics System
Purpose: Evaluates trained YOLO detector on held-out test or validation splits
         reporting exact scientific metrics (Precision, Recall, mAP@50, mAP@50-95).
         Never fabricates accuracy figures.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO


def evaluate_yolo_detector(
    model_path: str = "models/cargo_yolo_24class_best.pt",
    data_yaml: str = "deployment_dataset/data.yaml",
    split: str = "test",
    imgsz: int = 512,
    batch_size: int = 16,
    output_json: str = "results/evaluation/yolo_evaluation_report.json",
) -> Dict[str, Any]:
    """
    Evaluates YOLO model and writes evaluation metrics to JSON.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model weights not found at: {model_path}")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    print("=" * 78)
    print(f"CARGO VISION - YOLO OBJECT DETECTION EVALUATION (SPLIT: {split.upper()})")
    print("=" * 78)
    print(f"Model Weights: {model_path}")
    print(f"Dataset YAML:  {data_yaml}")
    print("=" * 78)

    model = YOLO(model_path)
    metrics = model.val(
        data=os.path.abspath(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch_size,
        verbose=True,
        workers=0,
    )

    p = float(metrics.results_dict.get("metrics/precision(B)", 0.0))
    r = float(metrics.results_dict.get("metrics/recall(B)", 0.0))
    map50 = float(metrics.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))

    report = {
        "model_path": os.path.abspath(model_path),
        "dataset_yaml": os.path.abspath(data_yaml),
        "split_evaluated": split,
        "classes_count": len(model.names) if hasattr(model, "names") else 24,
        "class_names": list(model.names.values()) if hasattr(model, "names") else [],
        "overall_metrics": {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
        },
    }

    with open(output_json, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78)
    print("EVALUATION RESULTS SUMMARY:")
    print("=" * 78)
    print(f"  - Precision (P):     {p:.4f} ({p*100:.2f}%)")
    print(f"  - Recall (R):        {r:.4f} ({r*100:.2f}%)")
    print(f"  - mAP @ 0.50:        {map50:.4f} ({map50*100:.2f}%)")
    print(f"  - mAP @ 0.50:0.95:   {map50_95:.4f} ({map50_95*100:.2f}%)")
    print("=" * 78)
    print(f"Saved evaluation report to: {output_json}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO detector on Cargo Vision dataset.")
    parser.add_argument("--model", type=str, default="models/cargo_yolo_24class_best.pt", help="Path to YOLO model.")
    parser.add_argument("--data", type=str, default="deployment_dataset/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "valid", "test"], help="Dataset split.")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--output", type=str, default="results/evaluation/yolo_evaluation_report.json", help="Output JSON report.")
    args = parser.parse_args()

    split_name = "test" if args.split in ["test"] else "val"
    evaluate_yolo_detector(
        model_path=args.model,
        data_yaml=args.data,
        split=split_name,
        imgsz=args.imgsz,
        batch_size=args.batch,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
