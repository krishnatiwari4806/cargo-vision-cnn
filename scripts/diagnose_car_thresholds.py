"""
diagnose_car_thresholds.py - Multi-Threshold Inference Diagnostic for data/car.jpg
==================================================================================
Project: Cargo Vision Logistics System
Purpose: Runs YOLOv8n Exp v2 on data/car.jpg across confidence thresholds:
         0.25, 0.15, 0.10, and 0.05.
         Generates annotated visualization images for each threshold.
"""

import os
import sys
import json
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO
import cv2
from PIL import Image

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cargo_yolo_exp_v2_best.pt")
IMAGE_PATH = os.path.join(PROJECT_ROOT, "data", "car.jpg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "inference")
THRESHOLDS = [0.25, 0.15, 0.10, 0.05]


def run_threshold_diagnostics():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)
    orig_img = Image.open(IMAGE_PATH)
    w_img, h_img = orig_img.size

    print("=" * 88)
    print(f"CARGO VISION - MULTI-THRESHOLD INFERENCE DIAGNOSTIC FOR: {IMAGE_PATH}")
    print(f"Image Size: {w_img} x {h_img} px | Model: {MODEL_PATH}")
    print("=" * 88)

    all_results = {}

    for conf in THRESHOLDS:
        res = model.predict(source=IMAGE_PATH, conf=conf, imgsz=640, verbose=False)[0]
        boxes = res.boxes

        detections = []
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                c_score = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = xyxy
                w_px = max(0.0, x2 - x1)
                h_px = max(0.0, y2 - y1)
                xc_norm = (x1 + w_px / 2.0) / w_img
                yc_norm = (y1 + h_px / 2.0) / h_img
                w_norm = w_px / w_img
                h_norm = h_px / h_img

                detections.append({
                    "id": i + 1,
                    "class_name": cls_name,
                    "class_id": cls_id,
                    "confidence": round(c_score, 4),
                    "confidence_pct": f"{c_score * 100:.2f}%",
                    "bbox_pixels": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "bbox_normalized": [round(xc_norm, 4), round(yc_norm, 4), round(w_norm, 4), round(h_norm, 4)],
                })

        detections.sort(key=lambda d: d["confidence"], reverse=True)

        # Save annotated image for this threshold
        conf_str = f"{conf:.2f}"
        annotated_path = os.path.join(OUTPUT_DIR, f"car_annotated_conf_{conf_str}.jpg")
        annotated_bgr = res.plot()
        cv2.imwrite(annotated_path, annotated_bgr)

        all_results[conf_str] = {
            "threshold": conf,
            "total_detections": len(detections),
            "annotated_image": annotated_path,
            "detections": detections,
        }

        print(f"\n>>> THRESHOLD = {conf:.2f} (Total Detections: {len(detections)})")
        print(f"    Annotated Image Saved: {annotated_path}")
        if detections:
            print(f"    {'#':<3} | {'Class Name':<14} | {'Confidence':<12} | {'Box Pixels [x1, y1, x2, y2]':<32} | {'Norm [xc, yc, w, h]'}")
            print("    " + "-" * 80)
            for d in detections:
                box_str = f"[{d['bbox_pixels'][0]}, {d['bbox_pixels'][1]}, {d['bbox_pixels'][2]}, {d['bbox_pixels'][3]}]"
                norm_str = f"[{d['bbox_normalized'][0]}, {d['bbox_normalized'][1]}, {d['bbox_normalized'][2]}, {d['bbox_normalized'][3]}]"
                print(f"    {d['id']:<3} | {d['class_name']:<14} | {d['confidence_pct']:<12} | {box_str:<32} | {norm_str}")
        else:
            print("    [No detections above this threshold]")

    print("\n" + "=" * 88)

    # Save summary report
    report_json_path = os.path.join(PROJECT_ROOT, "results", "inference", "car_threshold_diagnostic.json")
    with open(report_json_path, "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, indent=2)


if __name__ == "__main__":
    run_threshold_diagnostics()
