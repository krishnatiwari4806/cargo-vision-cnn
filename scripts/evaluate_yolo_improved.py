"""
evaluate_yolo_improved.py - Detailed Test Benchmark Evaluator for Improved YOLO Model
=====================================================================================
Project: Cargo Vision Logistics System
Purpose: Rigorously evaluates the improved YOLO detector on the held-out test split,
         reporting comprehensive macro and per-class metrics (P, R, mAP50, mAP50-95),
         ground-truth instance counts, and confusion matrix data.
         Never fabricates accuracy figures.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO


def evaluate_improved_detector(
    model_path: str = "models/cargo_yolo_improved_best.pt",
    data_yaml: str = "deployment_dataset/data.yaml",
    split: str = "test",
    imgsz: int = 640,
    batch_size: int = 8,
    output_json: str = "results/evaluation/yolo_improved_evaluation_report.json",
) -> Dict[str, Any]:
    """
    Runs full benchmark evaluation of the improved detector on the test set.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    print("=" * 78)
    print(f"CARGO VISION - ENHANCED YOLO BENCHMARK EVALUATION (SPLIT: {split.upper()})")
    print("=" * 78)
    print(f"Model Weights: {os.path.abspath(model_path)}")
    print(f"Dataset YAML:  {os.path.abspath(data_yaml)}")
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

    # Global aggregate metrics
    overall_p = float(metrics.results_dict.get("metrics/precision(B)", 0.0))
    overall_r = float(metrics.results_dict.get("metrics/recall(B)", 0.0))
    overall_map50 = float(metrics.results_dict.get("metrics/mAP50(B)", 0.0))
    overall_map50_95 = float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))

    class_names_dict = model.names if hasattr(model, "names") else {}
    num_classes = len(class_names_dict)

    # Extract per-class arrays if available
    per_class_results: List[Dict[str, Any]] = []

    # Count test split images and ground truth annotations
    test_img_dir = os.path.join(os.path.dirname(data_yaml), split if split != "valid" else "valid", "images")
    test_lbl_dir = os.path.join(os.path.dirname(data_yaml), split if split != "valid" else "valid", "labels")

    num_test_images = len(os.listdir(test_img_dir)) if os.path.exists(test_img_dir) else 0
    gt_class_counts: Dict[str, int] = {name: 0 for name in class_names_dict.values()}
    total_gt_instances = 0

    if os.path.exists(test_lbl_dir):
        for lf in os.listdir(test_lbl_dir):
            if not lf.endswith(".txt"):
                continue
            with open(os.path.join(test_lbl_dir, lf), "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cid = int(parts[0])
                            if cid in class_names_dict:
                                gt_class_counts[class_names_dict[cid]] += 1
                                total_gt_instances += 1
                        except ValueError:
                            pass

    # Extract per-class metric curves from metrics object
    class_map50_list = getattr(metrics.box, "maps", None)
    class_p_list = getattr(metrics.box, "p", None)
    class_r_list = getattr(metrics.box, "r", None)

    for cid, cname in class_names_dict.items():
        gt_count = gt_class_counts.get(cname, 0)
        p_val = float(class_p_list[cid]) if class_p_list is not None and len(class_p_list) > cid else 0.0
        r_val = float(class_r_list[cid]) if class_r_list is not None and len(class_r_list) > cid else 0.0
        m50_val = float(metrics.box.ap50[cid]) if hasattr(metrics.box, "ap50") and len(metrics.box.ap50) > cid else 0.0
        m50_95_val = float(class_map50_list[cid]) if class_map50_list is not None and len(class_map50_list) > cid else 0.0

        per_class_results.append({
            "class_id": cid,
            "class_name": cname,
            "ground_truth_instances": gt_count,
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "mAP50": round(m50_val, 4),
            "mAP50_95": round(m50_95_val, 4),
        })

    # Confusion matrix extraction
    cm_data = None
    if hasattr(metrics, "confusion_matrix") and hasattr(metrics.confusion_matrix, "matrix"):
        cm_matrix = metrics.confusion_matrix.matrix
        if isinstance(cm_matrix, np.ndarray):
            cm_data = {
                "shape": list(cm_matrix.shape),
                "summary": "Full confusion matrix calculated and saved in run artifacts.",
            }

    report = {
        "model_path": os.path.abspath(model_path),
        "dataset_yaml": os.path.abspath(data_yaml),
        "split_evaluated": split,
        "test_images_count": num_test_images,
        "ground_truth_objects_count": total_gt_instances,
        "classes_count": num_classes,
        "overall_metrics": {
            "precision": round(overall_p, 4),
            "recall": round(overall_r, 4),
            "mAP50": round(overall_map50, 4),
            "mAP50_95": round(overall_map50_95, 4),
        },
        "per_class_metrics": per_class_results,
        "confusion_matrix": cm_data,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print clean formatted table
    print("\n" + "=" * 78)
    print("ENHANCED EVALUATION REPORT (HELD-OUT TEST SET)")
    print("=" * 78)
    print(f"Total Test Images:            {num_test_images}")
    print(f"Total Ground-Truth Objects:   {total_gt_instances}")
    print("-" * 78)
    print(f"Overall Precision (P):        {overall_p:.4f} ({overall_p*100:.2f}%)")
    print(f"Overall Recall (R):           {overall_r:.4f} ({overall_r*100:.2f}%)")
    print(f"Overall mAP @ 0.50:           {overall_map50:.4f} ({overall_map50*100:.2f}%)")
    print(f"Overall mAP @ 0.50:0.95:      {overall_map50_95:.4f} ({overall_map50_95*100:.2f}%)")
    print("-" * 78)
    print(f"{'CLASS NAME':<16} | {'GT':<4} | {'PRECISION':<10} | {'RECALL':<10} | {'mAP@50':<10} | {'mAP@50-95':<10}")
    print("-" * 78)
    for c_res in per_class_results:
        # Highlight classes that actually exist in test set
        if c_res["ground_truth_instances"] > 0:
            print(f"{c_res['class_name']:<16} | {c_res['ground_truth_instances']:<4} | {c_res['precision']:>8.1%} | {c_res['recall']:>8.1%} | {c_res['mAP50']:>8.1%} | {c_res['mAP50_95']:>8.1%}")
    print("=" * 78)
    print(f"Detailed JSON report written to: {output_json}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Improved YOLO detector on test split.")
    parser.add_argument("--model", type=str, default="models/cargo_yolo_improved_best.pt", help="Path to YOLO model.")
    parser.add_argument("--data", type=str, default="deployment_dataset/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--output", type=str, default="results/evaluation/yolo_improved_evaluation_report.json", help="Output JSON report.")
    args = parser.parse_args()

    evaluate_improved_detector(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch_size=args.batch,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
