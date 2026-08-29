"""
predict_yolo.py - Object Detection & Localization Inference Pipeline (24 Classes)
==================================================================================
Project: Cargo Vision Logistics System
Purpose: Runs YOLOv8 object detection on arbitrary cargo images, localizing multiple
         objects and extracting bounding boxes and classification confidences across
         all 24 supported classes.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO


class YOLOCargoDetector:
    """
    YOLO Object Detector for multi-object cargo localization and classification.
    """

    def __init__(
        self,
        model_path: str = "models/cargo_yolo_24class_best.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._model = None

    def _get_model(self) -> YOLO:
        """Lazy loads YOLO model weights."""
        if self._model is None:
            if not os.path.exists(self.model_path):
                # Fallback to pretrained base model if custom model not yet trained
                fallback = "yolov8n.pt"
                if os.path.exists(fallback):
                    self.model_path = fallback
                else:
                    self.model_path = "yolov8n.pt"
            self._model = YOLO(self.model_path)
        return self._model

    def predict(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Runs object detection on the input image.
        
        Returns:
            Structured dictionary with all detected objects, bounding boxes, and confidences.
        """
        conf_thresh = conf if conf is not None else self.conf_threshold
        iou_thresh = iou if iou is not None else self.iou_threshold

        # Validate input if path
        img_path_str = None
        if isinstance(image_input, str):
            img_path_str = os.path.abspath(image_input)
            if not os.path.exists(img_path_str):
                return {
                    "status": "ERROR",
                    "error": f"Image file not found: '{image_input}'",
                    "image_path": image_input,
                    "detected_objects_count": 0,
                    "primary_detection": None,
                    "detections": [],
                }

        try:
            model = self._get_model()
            results = model.predict(
                source=image_input,
                conf=conf_thresh,
                iou=iou_thresh,
                verbose=False,
            )
        except Exception as e:
            return {
                "status": "ERROR",
                "error": f"YOLO prediction failed: {str(e)}",
                "image_path": str(image_input),
                "detected_objects_count": 0,
                "primary_detection": None,
                "detections": [],
            }

        res = results[0]
        boxes = res.boxes

        orig_h, orig_w = res.orig_shape if hasattr(res, "orig_shape") else (None, None)

        detections: List[Dict[str, Any]] = []

        if boxes is not None and len(boxes) > 0:
            names_dict = res.names
            xyxy_tensor = boxes.xyxy.cpu().numpy()
            conf_tensor = boxes.conf.cpu().numpy()
            cls_tensor = boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                cls_id = int(cls_tensor[i])
                class_name = names_dict.get(cls_id, f"class_{cls_id}")
                conf_val = float(conf_tensor[i])
                x1, y1, x2, y2 = [round(float(coord), 1) for coord in xyxy_tensor[i]]

                norm_box = None
                if orig_w and orig_h and orig_w > 0 and orig_h > 0:
                    norm_box = [
                        round(x1 / orig_w, 4),
                        round(y1 / orig_h, 4),
                        round(x2 / orig_w, 4),
                        round(y2 / orig_h, 4),
                    ]

                detections.append({
                    "class_name": class_name,
                    "class_id": cls_id,
                    "confidence": round(conf_val, 4),
                    "bbox_xyxy_px": [int(x1), int(y1), int(x2), int(y2)],
                    "bbox_normalized": norm_box,
                })

            # Sort by confidence descending
            detections.sort(key=lambda d: d["confidence"], reverse=True)

        primary_detection = detections[0] if len(detections) > 0 else None

        return {
            "status": "SUCCESS",
            "image_path": img_path_str if img_path_str else "in-memory",
            "image_dimensions_px": {"width": orig_w, "height": orig_h} if orig_w and orig_h else None,
            "detected_objects_count": len(detections),
            "primary_detection": primary_detection,
            "detections": detections,
        }


def main():
    parser = argparse.ArgumentParser(description="Run YOLO Object Detection on Cargo Images.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file.")
    parser.add_argument("--model", type=str, default="models/cargo_yolo_24class_best.pt", help="Path to YOLO weights.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45).")
    parser.add_argument("--json", action="store_true", help="Output raw JSON.")
    args = parser.parse_args()

    detector = YOLOCargoDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    result = detector.predict(args.image)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=" * 78)
    print("CARGO VISION - YOLO OBJECT DETECTION RESULTS (24 CLASSES)")
    print("=" * 78)
    print(f"Image Path:            {result.get('image_path')}")
    print(f"Status:                {result.get('status')}")
    print(f"Total Objects Found:   {result.get('detected_objects_count')}")
    print("-" * 78)

    if result.get("status") == "ERROR":
        print(f"[ERROR] {result.get('error')}")
        print("=" * 78)
        return

    if result["detected_objects_count"] == 0:
        print("[NOTICE] No objects detected at the given confidence threshold.")
        print("=" * 78)
        return

    print(f"{'INDEX':<6} | {'CLASS NAME':<16} | {'CONFIDENCE':<12} | {'BOUNDING BOX (X1, Y1, X2, Y2)':<30}")
    print("-" * 78)
    for idx, det in enumerate(result["detections"]):
        bb_str = f"[{det['bbox_xyxy_px'][0]}, {det['bbox_xyxy_px'][1]}, {det['bbox_xyxy_px'][2]}, {det['bbox_xyxy_px'][3]}]"
        print(f"{idx+1:<6} | {det['class_name']:<16} | {det['confidence']:>10.1%} | {bb_str:<30}")
    print("=" * 78)


if __name__ == "__main__":
    main()
