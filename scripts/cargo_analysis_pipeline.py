"""
cargo_analysis_pipeline.py - Unified Cargo Analysis & Multi-Load Fleet Dispatch Orchestrator
=============================================================================================
Project: Cargo Vision Logistics System
Purpose: Integrates and orchestrates the complete end-to-end cargo pipeline:
         1. Image Input Validation (Single & Multi-Load)
         2. Object Detection & Identification (YOLOv8 COCO detector with MobileNetV2 fallback)
         3. Physical Dimension Estimation (Reference-Marker / Category-Prior Fallback)
         4. Quantity & Multi-Load Volumetric Calculation (Stacking, Packing Factors, Floor Area)
         5. Multi-Constraint Vehicle Fleet Recommendation (VehicleDatabase)

Scientific & Accuracy Notice:
-----------------------------
1. No Synthetic Data: Uses real trained weights and physical geometric constraints.
2. Honest Accuracy Reporting: Exposes component-level confidence scores. The 95% target
   has not yet been validated against a real physical benchmark.
3. Explicit Warning Generation: Clearly flags prior-estimated dimensions, missing depth axes,
   and unverified payload weights.
"""

import os
import sys
import json
import math
import argparse
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import existing core modules
from scripts.vehicle_database import VehicleDatabase, VehicleSpec, vehicle_db
from scripts.dimension_estimator import DimensionEstimator, DimensionEstimateResult, EstimationStatus


# -----------------------------------------------------------------------------
# Configuration & Logistics Parameters
# -----------------------------------------------------------------------------

TARGET_CLASSES: List[str] = ["box", "chair", "couch", "suitcase", "table"]

# Stacking limit (max layers) per category
CATEGORY_STACKING_LIMITS: Dict[str, int] = {
    "box": 4,
    "chair": 2,
    "couch": 1,
    "table": 1,
    "suitcase": 3,
    "car": 1,
    "refrigerator": 1,
    "tv": 2,
    "bed": 1,
    "desk": 1,
}

# Packing efficiency factor (eta_pack) to account for stacking air gaps and packaging
CATEGORY_PACKING_FACTORS: Dict[str, float] = {
    "box": 0.80,
    "chair": 0.65,
    "couch": 0.55,
    "table": 0.60,
    "suitcase": 0.75,
    "car": 1.00,
    "refrigerator": 0.85,
    "tv": 0.80,
    "bed": 0.70,
    "desk": 0.65,
}

# Mapping from COCO detector classes to Cargo Vision canonical category priors
COCO_TO_CARGO_MAP: Dict[str, str] = {
    "car": "car",
    "chair": "chair",
    "couch": "couch",
    "sofa": "couch",
    "dining table": "table",
    "table": "table",
    "suitcase": "suitcase",
    "backpack": "suitcase",
    "handbag": "suitcase",
    "bed": "bed",
    "box": "box",
    "refrigerator": "refrigerator",
    "tv": "tv",
    "desk": "desk",
}


# -----------------------------------------------------------------------------
# Unified Pipeline Class
# -----------------------------------------------------------------------------

class CargoAnalysisPipeline:
    """
    End-to-End Cargo Vision Analysis and Fleet Dispatch Orchestrator.
    Supports:
      - Single Cargo Item Analysis
      - Multi-Load Shipment Aggregation
      - Dual vision backends (YOLOv8 COCO detector & MobileNetV2 classification fallback)
    """

    def __init__(
        self,
        model_path: str = "models/cargo_mobilenetv2_expanded_best.keras",
        yolo_model_path: str = "yolov8n.pt",
        custom_vehicle_db: Optional[VehicleDatabase] = None,
        custom_dimension_estimator: Optional[DimensionEstimator] = None,
        detector_backend: str = "auto",
        conf_threshold: float = 0.25,
    ):
        self.model_path = model_path
        self.yolo_model_path = yolo_model_path
        self.vehicle_db = custom_vehicle_db if custom_vehicle_db is not None else vehicle_db
        self.dimension_estimator = custom_dimension_estimator if custom_dimension_estimator is not None else DimensionEstimator()
        self.detector_backend = detector_backend
        self.conf_threshold = conf_threshold
        self._model = None       # Lazy-loaded MobileNetV2
        self._yolo_model = None  # Lazy-loaded YOLO

    def _get_model(self):
        """Lazy loads TensorFlow MobileNetV2 model to optimize startup time."""
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Trained model checkpoint not found at: {self.model_path}")
            import tensorflow as tf
            self._model = tf.keras.models.load_model(self.model_path)
        return self._model

    def _get_yolo_model(self):
        """Lazy loads Ultralytics YOLO model to optimize startup time."""
        if self._yolo_model is None:
            from ultralytics import YOLO
            self._yolo_model = YOLO(self.yolo_model_path)
        return self._yolo_model

    def validate_image_input(self, image_path: str) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
        """
        Validates image file existence and readability.
        """
        if not isinstance(image_path, str) or not image_path.strip():
            return False, "Image path must be a non-empty string.", None

        if not os.path.exists(image_path):
            return False, f"Image file not found: '{image_path}'", None

        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                img_rgb.load()
                return True, None, img_rgb
        except Exception as e:
            return False, f"Failed to decode image file '{image_path}': {str(e)}", None

    def detect_with_yolo(
        self,
        image_input: Union[str, Image.Image],
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Executes object detection using YOLOv8 COCO model.
        Returns detection summary dictionary and warnings, or None if no valid objects found.
        """
        warnings = []
        try:
            model = self._get_yolo_model()
            results = model.predict(source=image_input, conf=self.conf_threshold, verbose=False)[0]
            boxes = results.boxes

            if boxes is None or len(boxes) == 0:
                return None, ["No objects detected by YOLO detector above confidence threshold."]

            detected_items = []
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                raw_cls_name = model.names.get(cls_id, f"class_{cls_id}").lower()
                c_score = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist()

                canonical_class = COCO_TO_CARGO_MAP.get(raw_cls_name, raw_cls_name)

                detected_items.append({
                    "raw_class": raw_cls_name,
                    "canonical_class": canonical_class,
                    "confidence": round(c_score, 4),
                    "bbox_px": [round(x, 1) for x in xyxy],
                })

            if not detected_items:
                return None, ["No supported cargo objects found in YOLO detections."]

            # Prioritize highest confidence detection
            detected_items.sort(key=lambda d: d["confidence"], reverse=True)
            primary = detected_items[0]

            # Count instances matching primary class
            primary_instances = [d for d in detected_items if d["canonical_class"] == primary["canonical_class"]]

            detection_res = {
                "class_name": primary["canonical_class"],
                "confidence": primary["confidence"],
                "raw_class_name": primary["raw_class"],
                "detector_backend": "yolo_coco",
                "bounding_box_px": primary["bbox_px"],
                "detected_instance_count": len(primary_instances),
                "all_detections": detected_items,
                "probabilities": {primary["canonical_class"]: primary["confidence"]},
            }

            return detection_res, warnings
        except Exception as e:
            warnings.append(f"YOLO detection encountered error: {str(e)}")
            return None, warnings

    def classify_image(self, img_rgb: Image.Image) -> Tuple[Dict[str, Any], List[str]]:
        """
        Executes object classification using MobileNetV2 expanded model.
        """
        warnings = []
        model = self._get_model()

        # Preprocess: resize to 128x128 and convert to float32 batch
        img_resized = img_rgb.resize((128, 128), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32)
        batch = np.expand_dims(img_array, axis=0)

        probs = model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = TARGET_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        probabilities = {
            TARGET_CLASSES[i]: round(float(probs[i]), 4)
            for i in range(len(TARGET_CLASSES))
        }

        if confidence < 0.50:
            warnings.append(
                f"Classification confidence ({confidence:.1%}) is low. "
                "Image may be ambiguous, poorly lit, or out-of-distribution."
            )

        classification_res = {
            "class_name": pred_class,
            "confidence": round(confidence, 4),
            "detector_backend": "mobilenetv2",
            "probabilities": probabilities,
        }

        return classification_res, warnings

    def calculate_cargo_requirements(
        self,
        dimensions: Dict[str, Optional[float]],
        category: str,
        quantity: int,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Calculates unit volume, required total volume, stacking floor area, and cargo footprint.
        """
        warnings = []
        length = dimensions.get("length_cm")
        width = dimensions.get("width_cm")
        height = dimensions.get("height_cm")

        stacking_limit = CATEGORY_STACKING_LIMITS.get(category.lower(), 1)
        packing_factor = CATEGORY_PACKING_FACTORS.get(category.lower(), 0.70)

        unit_vol_m3 = None
        total_vol_m3 = None
        req_floor_area_m2 = None

        if length is not None and width is not None and height is not None:
            # All 3 dimensions are available
            unit_vol_m3 = round((length * width * height) / 1000000.0, 4)
            raw_total_vol = unit_vol_m3 * quantity
            total_vol_m3 = round(raw_total_vol / packing_factor, 4)

            # Stacking and floor area calculation
            stack_layers = min(quantity, stacking_limit)
            footprint_items = math.ceil(quantity / stack_layers)
            req_floor_area_m2 = round(((length * width) / 10000.0) * footprint_items, 4)
        else:
            # Incomplete dimensions
            warnings.append(
                "Total cargo volume and floor area cannot be computed because one or more dimensions (e.g. depth/width) are unmeasured."
            )

        cargo_summary = {
            "total_items": quantity,
            "unit_volume_m3": unit_vol_m3,
            "total_volume_m3": total_vol_m3,
            "stacking_limit_layers": stacking_limit,
            "required_floor_area_m2": req_floor_area_m2,
        }

        return cargo_summary, warnings

    def recommend_vehicle(
        self,
        category: str,
        dimensions: Dict[str, Optional[float]],
        cargo_summary: Dict[str, Any],
        quantity: int,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Matches single-category cargo requirements against the structured Vehicle Database using hard constraints.
        """
        warnings = []
        warnings.append("Payload suitability cannot be verified because physical object weight is unavailable.")

        length = dimensions.get("length_cm")
        width = dimensions.get("width_cm")
        height = dimensions.get("height_cm")
        total_vol = cargo_summary.get("total_volume_m3")
        req_floor_area = cargo_summary.get("required_floor_area_m2")

        all_vehicles = self.vehicle_db.get_all_vehicles()
        suitable_vehicles: List[VehicleSpec] = []

        for v in all_vehicles:
            # Constraint 1: Single item physical dimension fit
            if length is not None and length > v.usable_length_cm:
                continue
            if width is not None and width > v.usable_width_cm:
                continue
            if height is not None and height > v.usable_height_cm:
                continue

            # Constraint 2: Total required volume fit
            if total_vol is not None and total_vol > v.usable_volume_m3:
                continue

            # Constraint 3: Required floor area fit
            if req_floor_area is not None and req_floor_area > v.floor_area_m2:
                continue

            suitable_vehicles.append(v)

        if not suitable_vehicles:
            # Fallback: cargo exceeds single standard vehicle capacity
            largest_v = all_vehicles[-1]
            return {
                "vehicle_id": largest_v.vehicle_id,
                "vehicle_name": largest_v.vehicle_name,
                "reason": f"Cargo requirement ({quantity} {category}s) exceeds standard single-vehicle capacity. Multi-trip or fleet batching using {largest_v.vehicle_name} is required.",
                "alternatives": [],
            }, warnings

        # Select smallest fitting vehicle tier (natural capacity ordering)
        primary_v = suitable_vehicles[0]
        alternatives = [v.vehicle_name for v in suitable_vehicles[1:3]]

        # Construct clear reasoning
        reason_parts = [f"Accommodates {quantity} {category}(s)"]
        if total_vol is not None:
            reason_parts.append(f"requiring {total_vol:.2f} m³ usable space (vehicle capacity: {primary_v.usable_volume_m3:.2f} m³)")
        if req_floor_area is not None:
            reason_parts.append(f"and {req_floor_area:.2f} m² floor bed area (vehicle bed: {primary_v.floor_area_m2:.2f} m²)")
        
        reason_str = ", ".join(reason_parts) + "."

        recommendation = {
            "vehicle_id": primary_v.vehicle_id,
            "vehicle_name": primary_v.vehicle_name,
            "reason": reason_str,
            "alternatives": alternatives,
        }

        return recommendation, warnings

    def recommend_vehicle_for_shipment(
        self,
        items: List[Dict[str, Any]],
        total_volume_m3: Optional[float],
        total_floor_area_m2: Optional[float],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Matches multi-item combined cargo shipment requirements against the Vehicle Database.
        """
        warnings = []
        warnings.append("Payload suitability cannot be verified because physical object weight is unavailable.")

        all_vehicles = self.vehicle_db.get_all_vehicles()
        suitable_vehicles: List[VehicleSpec] = []

        for v in all_vehicles:
            # Constraint 1: Every individual item's dimensions must fit inside vehicle
            fits_dimensions = True
            for item in items:
                dims = item.get("dimensions", {})
                l = dims.get("length_cm")
                w = dims.get("width_cm")
                h = dims.get("height_cm")
                if l is not None and l > v.usable_length_cm:
                    fits_dimensions = False
                    break
                if w is not None and w > v.usable_width_cm:
                    fits_dimensions = False
                    break
                if h is not None and h > v.usable_height_cm:
                    fits_dimensions = False
                    break
            if not fits_dimensions:
                continue

            # Constraint 2: Total required volume fit
            if total_volume_m3 is not None and total_volume_m3 > v.usable_volume_m3:
                continue

            # Constraint 3: Total required floor area fit
            if total_floor_area_m2 is not None and total_floor_area_m2 > v.floor_area_m2:
                continue

            suitable_vehicles.append(v)

        total_quantity = sum(item.get("quantity", 1) for item in items)
        category_summary_str = ", ".join(f"{item.get('quantity', 1)} {item.get('category', 'item')}(s)" for item in items)

        if not suitable_vehicles:
            largest_v = all_vehicles[-1]
            return {
                "vehicle_id": largest_v.vehicle_id,
                "vehicle_name": largest_v.vehicle_name,
                "reason": (
                    f"Combined shipment ({total_quantity} items: {category_summary_str}) exceeds "
                    f"standard single-vehicle capacity. Multi-trip or fleet batching using {largest_v.vehicle_name} is required."
                ),
                "alternatives": [],
            }, warnings

        # Select smallest fitting vehicle tier
        primary_v = suitable_vehicles[0]
        alternatives = [v.vehicle_name for v in suitable_vehicles[1:3]]

        reason_parts = [f"Accommodates combined shipment ({total_quantity} items: {category_summary_str})"]
        if total_volume_m3 is not None:
            reason_parts.append(f"requiring {total_volume_m3:.2f} m³ usable space (vehicle capacity: {primary_v.usable_volume_m3:.2f} m³)")
        if total_floor_area_m2 is not None:
            reason_parts.append(f"and {total_floor_area_m2:.2f} m² floor bed area (vehicle bed: {primary_v.floor_area_m2:.2f} m²)")

        reason_str = ", ".join(reason_parts) + "."

        return {
            "vehicle_id": primary_v.vehicle_id,
            "vehicle_name": primary_v.vehicle_name,
            "reason": reason_str,
            "alternatives": alternatives,
        }, warnings

    def analyze(
        self,
        image_path: str,
        quantity: int = 1,
        known_marker_size_cm: Optional[float] = None,
        object_bbox_px: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end analysis on a single cargo photo.
        """
        all_warnings = ["95% target has not yet been validated against a measured physical benchmark."]

        # 1. Validate Quantity
        if not isinstance(quantity, int) or quantity < 1:
            return {
                "status": "ERROR",
                "error": f"Quantity must be a positive integer (>= 1), got {quantity}",
                "input_image": image_path,
                "warnings": all_warnings,
            }

        # 2. Validate Image Input
        valid, err_msg, img_rgb = self.validate_image_input(image_path)
        if not valid or img_rgb is None:
            return {
                "status": "ERROR",
                "error": err_msg,
                "input_image": image_path,
                "warnings": all_warnings,
            }

        # 3. Object Identification (YOLO COCO detection with MobileNetV2 fallback)
        classification_res = None
        detected_category = None
        auto_bbox = None

        if self.detector_backend in ("auto", "yolo"):
            yolo_res, yolo_warnings = self.detect_with_yolo(image_path)
            if yolo_res is not None:
                all_warnings.extend(yolo_warnings)
                classification_res = yolo_res
                detected_category = yolo_res["class_name"]
                if object_bbox_px is None and "bounding_box_px" in yolo_res:
                    auto_bbox = tuple(int(round(coord)) for coord in yolo_res["bounding_box_px"])

        # Fallback to MobileNetV2 if YOLO did not detect any object or backend is mobilenetv2
        if classification_res is None:
            if self.detector_backend == "yolo":
                all_warnings.extend(yolo_warnings)
                return {
                    "status": "ERROR",
                    "error": "No objects detected by YOLO detector above threshold.",
                    "input_image": image_path,
                    "warnings": all_warnings,
                }
            try:
                classification_res, cls_warnings = self.classify_image(img_rgb)
                all_warnings.extend(cls_warnings)
                detected_category = classification_res["class_name"]
            except Exception as e:
                return {
                    "status": "ERROR",
                    "error": f"Classification model execution failed: {str(e)}",
                    "input_image": image_path,
                    "warnings": all_warnings,
                }

        # 4. Dimension Estimation
        effective_bbox = object_bbox_px if object_bbox_px is not None else auto_bbox
        dim_res = self.dimension_estimator.estimate(
            image_input=image_path if known_marker_size_cm is not None and effective_bbox is not None else None,
            category=detected_category,
            known_marker_size_cm=known_marker_size_cm,
            object_bbox_px=effective_bbox,
            fallback_to_prior=True,
        )
        all_warnings.extend(dim_res.warnings)

        dimensions_dict = {
            "status": dim_res.status,
            "source": dim_res.source,
            "length_cm": dim_res.dimensions_cm.get("length"),
            "width_cm": dim_res.dimensions_cm.get("width"),
            "height_cm": dim_res.dimensions_cm.get("height"),
            "confidence": dim_res.confidence,
        }

        # 5. Total Cargo Calculation
        cargo_summary, cargo_warnings = self.calculate_cargo_requirements(
            dimensions=dimensions_dict,
            category=detected_category,
            quantity=quantity,
        )
        all_warnings.extend(cargo_warnings)

        # 6. Vehicle Recommendation
        vehicle_rec, veh_warnings = self.recommend_vehicle(
            category=detected_category,
            dimensions=dimensions_dict,
            cargo_summary=cargo_summary,
            quantity=quantity,
        )
        all_warnings.extend(veh_warnings)

        # Deduplicate warnings while preserving order
        seen_w = set()
        deduped_warnings = []
        for w in all_warnings:
            if w not in seen_w:
                seen_w.add(w)
                deduped_warnings.append(w)

        # Final Result Schema
        return {
            "status": "SUCCESS",
            "mode": "SINGLE_LOAD",
            "input_image": os.path.abspath(image_path),
            "classification": classification_res,
            "dimensions": dimensions_dict,
            "quantity": quantity,
            "cargo_summary": cargo_summary,
            "vehicle_recommendation": vehicle_rec,
            "warnings": deduped_warnings,
        }

    def analyze_multiple(
        self,
        image_paths: List[str],
        quantities: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-load analysis on multiple cargo photos.
        Aggregates items into a single unified shipment dispatch recommendation.
        """
        if not image_paths or not isinstance(image_paths, list):
            return {
                "status": "ERROR",
                "error": "image_paths must be a non-empty list of image file paths.",
                "warnings": ["95% target has not yet been validated against a measured physical benchmark."],
            }

        if quantities is None:
            quantities = [1] * len(image_paths)
        elif len(quantities) != len(image_paths):
            return {
                "status": "ERROR",
                "error": f"Length of quantities ({len(quantities)}) must match length of image_paths ({len(image_paths)}).",
                "warnings": ["95% target has not yet been validated against a measured physical benchmark."],
            }

        shipment_items = []
        all_warnings = ["95% target has not yet been validated against a measured physical benchmark."]

        for idx, (img_path, qty) in enumerate(zip(image_paths, quantities)):
            single_res = self.analyze(image_path=img_path, quantity=qty)
            if single_res.get("status") == "ERROR":
                return {
                    "status": "ERROR",
                    "error": f"Failed analyzing image #{idx+1} '{img_path}': {single_res.get('error')}",
                    "warnings": all_warnings,
                }
            all_warnings.extend(single_res.get("warnings", []))
            item_entry = {
                "item_index": idx + 1,
                "input_image": single_res["input_image"],
                "category": single_res["classification"]["class_name"],
                "classification": single_res["classification"],
                "quantity": qty,
                "dimensions": single_res["dimensions"],
                "cargo_summary": single_res["cargo_summary"],
            }
            shipment_items.append(item_entry)

        # Volumetric & Space Aggregation
        total_items = sum(item["quantity"] for item in shipment_items)
        volumes = [item["cargo_summary"]["total_volume_m3"] for item in shipment_items]
        floor_areas = [item["cargo_summary"]["required_floor_area_m2"] for item in shipment_items]

        total_volume_m3 = round(sum(v for v in volumes if v is not None), 4) if all(v is not None for v in volumes) else None
        total_floor_area_m2 = round(sum(a for a in floor_areas if a is not None), 4) if all(a is not None for a in floor_areas) else None

        shipment_summary = {
            "total_items": total_items,
            "total_volume_m3": total_volume_m3,
            "total_floor_area_m2": total_floor_area_m2,
            "item_breakdown": [
                {
                    "category": item["category"],
                    "quantity": item["quantity"],
                    "unit_volume_m3": item["cargo_summary"]["unit_volume_m3"],
                    "total_volume_m3": item["cargo_summary"]["total_volume_m3"],
                    "required_floor_area_m2": item["cargo_summary"]["required_floor_area_m2"],
                    "dimensions_cm": {
                        "length": item["dimensions"]["length_cm"],
                        "width": item["dimensions"]["width_cm"],
                        "height": item["dimensions"]["height_cm"],
                    },
                }
                for item in shipment_items
            ],
        }

        # Multi-Load Vehicle Recommendation
        vehicle_rec, veh_warnings = self.recommend_vehicle_for_shipment(
            items=shipment_items,
            total_volume_m3=total_volume_m3,
            total_floor_area_m2=total_floor_area_m2,
        )
        all_warnings.extend(veh_warnings)

        # Deduplicate warnings
        seen_w = set()
        deduped_warnings = []
        for w in all_warnings:
            if w not in seen_w:
                seen_w.add(w)
                deduped_warnings.append(w)

        return {
            "status": "SUCCESS",
            "mode": "MULTI_LOAD",
            "total_items": total_items,
            "items": shipment_items,
            "shipment_summary": shipment_summary,
            "vehicle_recommendation": vehicle_rec,
            "warnings": deduped_warnings,
        }


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cargo Vision End-to-End Logistics Pipeline CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to a single cargo image file.")
    parser.add_argument("--images", type=str, nargs="+", default=None, help="Paths to multiple cargo image files for multi-load aggregation.")
    parser.add_argument("--quantity", type=int, default=1, help="Quantity of items for single image mode (default: 1).")
    parser.add_argument("--quantities", type=int, nargs="+", default=None, help="Quantities corresponding to --images list.")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "yolo", "mobilenetv2"],
                        help="Vision backend: 'auto' (YOLO with MobileNetV2 fallback), 'yolo', or 'mobilenetv2'.")
    parser.add_argument("--marker-size", type=float, default=None, help="Optional known physical marker size in cm.")
    parser.add_argument("--bbox", type=int, nargs=4, default=None, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        help="Optional object bounding box in pixels (xmin ymin xmax ymax).")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted text.")
    args = parser.parse_args()

    if not args.image and not args.images:
        parser.error("Either --image <path> or --images <path1> <path2> ... must be provided.")

    pipeline = CargoAnalysisPipeline(detector_backend=args.backend)

    # Multi-Image Mode
    if args.images:
        result = pipeline.analyze_multiple(
            image_paths=args.images,
            quantities=args.quantities,
        )

        if args.json:
            print(json.dumps(result, indent=2))
            return

        if result.get("status") == "ERROR":
            print("\n" + "=" * 78)
            print("CARGO VISION MULTI-LOAD ANALYSIS ERROR")
            print("=" * 78)
            print(f"Error: {result.get('error')}")
            print("=" * 78)
            return

        rec = result["vehicle_recommendation"]
        summary = result["shipment_summary"]

        print("\n" + "=" * 78)
        print("CARGO VISION - MULTI-LOAD DISPATCH REPORT")
        print("=" * 78)
        print("\nSHIPMENT ITEMS")
        print("-" * 78)
        for idx, item in enumerate(result["items"]):
            cat = item["category"].capitalize()
            qty = item["quantity"]
            dims = item["dimensions"]
            c_sum = item["cargo_summary"]
            dim_str = f"{dims['length_cm']} x {dims['width_cm']} x {dims['height_cm']} cm" if dims['length_cm'] else "Unmeasured"
            vol_str = f"{c_sum['total_volume_m3']:.4f} m³" if c_sum['total_volume_m3'] else "N/A"
            print(f"{idx+1:>2}. {cat:<16} x{qty:<3} (Unit: {dim_str}, Total Vol: {vol_str})")

        print(f"\nTOTAL ITEMS:      {result['total_items']}")
        print(f"TOTAL VOLUME:     {summary['total_volume_m3']:.4f} m³" if summary['total_volume_m3'] else "TOTAL VOLUME:     N/A")
        print(f"TOTAL FLOOR AREA: {summary['total_floor_area_m2']:.4f} m²" if summary['total_floor_area_m2'] else "TOTAL FLOOR AREA: N/A")
        print("-" * 78)
        print("RECOMMENDED VEHICLE")
        print("-" * 78)
        print(f"Vehicle:          {rec['vehicle_name']}")
        print(f"Vehicle ID:       {rec['vehicle_id']}")
        print(f"Reason:           {rec['reason']}")
        if rec.get("alternatives"):
            print(f"Alternatives:     {', '.join(rec['alternatives'])}")
        print("-" * 78)
        print("IMPORTANT NOTICES & DISCLAIMERS:")
        for w in result["warnings"]:
            print(f"  * {w}")
        print("=" * 78)
        return

    # Single-Image Mode (Preserved exactly)
    result = pipeline.analyze(
        image_path=args.image,
        quantity=args.quantity,
        known_marker_size_cm=args.marker_size,
        object_bbox_px=tuple(args.bbox) if args.bbox else None,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    if result.get("status") == "ERROR":
        print("\n" + "=" * 78)
        print("CARGO VISION ANALYSIS ERROR")
        print("=" * 78)
        print(f"Error: {result.get('error')}")
        print(f"File:  {result.get('input_image')}")
        print("=" * 78)
        return

    cls_info = result["classification"]
    dim_info = result["dimensions"]
    summary = result["cargo_summary"]
    rec = result["vehicle_recommendation"]

    print("\n" + "=" * 78)
    print("CARGO VISION LOGISTICS DISPATCH REPORT")
    print("=" * 78)
    print(f"Input Image:           {result['input_image']}")
    print(f"Detected Cargo:        {cls_info['class_name'].upper()} (Confidence: {cls_info['confidence']:.1%})")
    if "bounding_box_px" in cls_info:
        print(f"Bounding Box:          {cls_info['bounding_box_px']} px")
    print(f"Quantity:              {result['quantity']} item(s)")
    print("-" * 78)
    print("PHYSICAL DIMENSION ESTIMATE:")
    print(f"  - Status:            {dim_info['status']} ({dim_info['source']})")
    print(f"  - Length:            {dim_info['length_cm']:.1f} cm" if dim_info['length_cm'] is not None else "  - Length:            None")
    print(f"  - Width (Depth):     {dim_info['width_cm']:.1f} cm" if dim_info['width_cm'] is not None else "  - Width (Depth):     None")
    print(f"  - Height:            {dim_info['height_cm']:.1f} cm" if dim_info['height_cm'] is not None else "  - Height:            None")
    print(f"  - Dimension Conf:    {dim_info['confidence']:.1%}")
    print("-" * 78)
    print("CARGO VOLUME & SPACE REQUIREMENTS:")
    print(f"  - Unit Volume:       {summary['unit_volume_m3']} m³" if summary['unit_volume_m3'] is not None else "  - Unit Volume:       None")
    print(f"  - Total Req Volume:  {summary['total_volume_m3']} m³ (adjusted for packing)" if summary['total_volume_m3'] is not None else "  - Total Req Volume:  None")
    print(f"  - Floor Area Bed:    {summary['required_floor_area_m2']} m² (max {summary['stacking_limit_layers']} stack layers)" if summary['required_floor_area_m2'] is not None else "  - Floor Area Bed:    None")
    print("-" * 78)
    print("FLEET RECOMMENDATION:")
    print(f"  - Recommended:       {rec['vehicle_name']}")
    print(f"  - Vehicle ID:        {rec['vehicle_id']}")
    print(f"  - Reason:            {rec['reason']}")
    if rec.get("alternatives"):
        print(f"  - Alternatives:      {', '.join(rec['alternatives'])}")
    print("-" * 78)
    print("IMPORTANT NOTICES & DISCLAIMERS:")
    for w in result["warnings"]:
        print(f"  * {w}")
    print("=" * 78)


if __name__ == "__main__":
    main()