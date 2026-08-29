"""
test_exp_v2_inference.py - Isolated End-to-End Inference Tester for YOLOv8n Exp v2
===================================================================================
Project: Cargo Vision Logistics System
Purpose: Runs object detection inference on a user-provided image using the trained
         models/cargo_yolo_exp_v2_best.pt checkpoint, prints detailed detection results,
         and saves an annotated copy of the image with bounding boxes.
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO
import cv2
from PIL import Image

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cargo_yolo_exp_v2_best.pt")
DEFAULT_TEST_IMAGE = os.path.join(
    PROJECT_ROOT, "deployment_dataset_expanded", "test", "images", "orig_train_Chair_16_JPG.rf.0261cecaf9b0251cdfb8d7cdf50d7988.jpg"
)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "inference")


def run_detection_inference(
    image_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    conf_threshold: float = 0.25,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    """
    Loads YOLOv8n Exp v2 model, detects cargo objects in image_path, prints details,
    and saves the annotated output image.
    """
    image_path = os.path.abspath(image_path)
    model_path = os.path.abspath(model_path)
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {model_path}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load trained YOLO model
    model = YOLO(model_path)

    # 2. Run inference
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        imgsz=640,
        verbose=False,
    )

    result = results[0]
    boxes = result.boxes

    # Get image dimensions
    orig_img = Image.open(image_path)
    img_width, img_height = orig_img.size

    detections: List[Dict[str, Any]] = []

    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            cls_name = model.names.get(cls_id, f"class_{cls_id}")
            conf = float(boxes.conf[i].item())

            # Bounding box coordinates (xyxy in pixels, and xywh normalized)
            xyxy = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = xyxy
            w_px = max(0.0, x2 - x1)
            h_px = max(0.0, y2 - y1)
            xc_norm = (x1 + w_px / 2.0) / img_width
            yc_norm = (y1 + h_px / 2.0) / img_height
            w_norm = w_px / img_width
            h_norm = h_px / img_height

            detections.append({
                "detection_id": i + 1,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "confidence_percent": f"{conf * 100:.2f}%",
                "bbox_pixels": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                    "width": round(w_px, 1),
                    "height": round(h_px, 1),
                },
                "bbox_normalized": {
                    "x_center": round(xc_norm, 4),
                    "y_center": round(yc_norm, 4),
                    "width": round(w_norm, 4),
                    "height": round(h_norm, 4),
                },
            })

    # Sort detections by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # 3. Save annotated image with bounding boxes
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    annotated_filename = f"annotated_{base_name}.jpg"
    annotated_path = os.path.join(output_dir, annotated_filename)

    # Use Ultralytics plotting for clean bounding box visualization
    annotated_img_bgr = result.plot()
    cv2.imwrite(annotated_path, annotated_img_bgr)

    # Print formatted detection report to terminal
    print("=" * 84)
    print("CARGO VISION - YOLOv8n EXP V2 INFERENCE TEST REPORT")
    print("=" * 84)
    print(f"Input Image:         {image_path}")
    print(f"Image Resolution:    {img_width} x {img_height} px")
    print(f"Model Checkpoint:    {model_path}")
    print(f"Confidence Threshold:{conf_threshold}")
    print(f"Total Objects Found: {len(detections)}")
    print(f"Annotated Image:     {annotated_path}")
    print("-" * 84)

    if detections:
        print(f"{'#':<3} | {'Class Name':<16} | {'Confidence':<12} | {'Box (x1, y1, x2, y2)':<28} | {'Normalized (w, h)'}")
        print("-" * 84)
        for d in detections:
            box_px = f"[{d['bbox_pixels']['x1']}, {d['bbox_pixels']['y1']}, {d['bbox_pixels']['x2']}, {d['bbox_pixels']['y2']}]"
            norm_dim = f"[{d['bbox_normalized']['width']}, {d['bbox_normalized']['height']}]"
            print(f"{d['detection_id']:<3} | {d['class_name']:<16} | {d['confidence_percent']:<12} | {box_px:<28} | {norm_dim}")
    else:
        print("No objects detected above the confidence threshold.")

    print("=" * 84)

    return {
        "input_image": image_path,
        "model_checkpoint": model_path,
        "image_dimensions": {"width": img_width, "height": img_height},
        "total_detections": len(detections),
        "annotated_image_path": annotated_path,
        "detections": detections,
    }


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8n Exp v2 object detection inference on an image.")
    parser.add_argument("--image", type=str, default=DEFAULT_TEST_IMAGE, help="Path to input image")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to model checkpoint")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save annotated image")
    parser.add_argument("--json", action="store_true", help="Print structured JSON output")
    args = parser.parse_args()

    result = run_detection_inference(
        image_path=args.image,
        model_path=args.model,
        conf_threshold=args.conf,
        output_dir=args.output_dir,
    )

    if args.json:
        print("\nJSON OUTPUT:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
