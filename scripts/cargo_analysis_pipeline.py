"""
cargo_analysis_pipeline.py - Unified Cargo Analysis & Multi-Load Fleet Dispatch Orchestrator
=============================================================================================
Project: Cargo Vision Logistics System
Purpose: Integrates and orchestrates the complete end-to-end cargo pipeline:
         1. Image Input Validation (Single, Multi-Image, and In-Image Multi-Object Extraction)
         2. Object Detection & Identification (YOLOv8 COCO detector with MobileNetV2 fallback)
         3. Physical Dimension & Payload Weight Estimation (Category Priors & Marker Measurement)
         4. Quantity & Multi-Load Volumetric & Weight Calculation (Stacking, Packing Factors, Floor Area)
         5. Multi-Constraint Vehicle Fleet Recommendation (Dimensions, Volume, Floor Area, Payload Capacity)

Scientific & Accuracy Notice:
-----------------------------
1. No Synthetic Data: Uses real trained weights and physical geometric/weight constraints.
2. Honest Accuracy Reporting: Exposes component-level confidence scores. The 95% target
   has not yet been validated against a real physical benchmark.
3. Explicit Warning Generation: Clearly flags prior-estimated dimensions/weights, missing depth axes,
   and unverified physical scale weights.
"""

import os
import sys
import json
import math
import argparse
import textwrap
import itertools
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

# Categories requiring vertical/upright transport orientation
UPRIGHT_CARGO_CATEGORIES: set = {
    "refrigerator",
    "tv",
    "car",
    "chair",
    "table",
    "couch",
    "bed",
    "desk",
}

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
# Reliability and Confidence Policy Thresholds
# -----------------------------------------------------------------------------
# Detections strictly below MIN_CONFIDENCE_THRESHOLD are rejected as detector noise
MIN_CONFIDENCE_THRESHOLD: float = 0.25

# Detections at or above RELIABLE_CONFIDENCE_THRESHOLD are considered reliable for automated dispatch
RELIABLE_CONFIDENCE_THRESHOLD: float = 0.50



# -----------------------------------------------------------------------------
# 3D Cargo Dimension & Physical Fit Helpers
# -----------------------------------------------------------------------------

def _dimensions_fit_vehicle(
    cargo_dimensions: Union[Dict[str, Optional[float]], Tuple[Optional[float], ...], List[Optional[float]]],
    vehicle_dimensions: Any,
    allow_rotation: bool = True,
    allow_3d_rotation: Optional[bool] = None,
    category: Optional[str] = None,
) -> bool:
    """
    Validates whether individual 3D cargo physical dimensions fit inside a vehicle's usable cargo hold.

    Args:
        cargo_dimensions: Dict with 'length_cm', 'width_cm', 'height_cm' (or tuple/list of dimensions in cm).
        vehicle_dimensions: VehicleSpec instance or dict/tuple with usable dimensions in cm.
        allow_rotation: If True, allows valid physical re-orientation of cargo inside the vehicle.
        allow_3d_rotation: If True, allows all 6 3D permutations (pitch/roll/yaw).
                           If False, allows only 2D yaw rotation (L <-> W) on vehicle bed.
                           If None (default), automatically determined by category (upright vs unconstrained).
        category: Optional cargo category name to check upright constraints.

    Returns:
        bool: True if cargo fits within vehicle dimensions in at least one valid orientation; False otherwise.
    """
    # 1. Extract cargo dimensions
    if isinstance(cargo_dimensions, dict):
        l = cargo_dimensions.get("length_cm")
        w = cargo_dimensions.get("width_cm")
        h = cargo_dimensions.get("height_cm")
    elif isinstance(cargo_dimensions, (list, tuple)):
        l = cargo_dimensions[0] if len(cargo_dimensions) > 0 else None
        w = cargo_dimensions[1] if len(cargo_dimensions) > 1 else None
        h = cargo_dimensions[2] if len(cargo_dimensions) > 2 else None
    else:
        return True

    # 2. Extract vehicle usable dimensions
    if hasattr(vehicle_dimensions, "usable_length_cm"):
        vl = float(vehicle_dimensions.usable_length_cm)
        vw = float(vehicle_dimensions.usable_width_cm)
        vh = float(vehicle_dimensions.usable_height_cm)
    elif isinstance(vehicle_dimensions, dict):
        vl = float(vehicle_dimensions.get("usable_length_cm") or vehicle_dimensions.get("length_cm", 0.0))
        vw = float(vehicle_dimensions.get("usable_width_cm") or vehicle_dimensions.get("width_cm", 0.0))
        vh = float(vehicle_dimensions.get("usable_height_cm") or vehicle_dimensions.get("height_cm", 0.0))
    elif isinstance(vehicle_dimensions, (list, tuple)) and len(vehicle_dimensions) >= 3:
        vl = float(vehicle_dimensions[0])
        vw = float(vehicle_dimensions[1])
        vh = float(vehicle_dimensions[2])
    else:
        return True

    if vl <= 0 or vw <= 0 or vh <= 0:
        return False

    # 3. No dimensions provided to constrain
    if l is None and w is None and h is None:
        return True

    # 4. Strict exact orientation check (no rotation)
    if not allow_rotation:
        if l is not None and l > vl:
            return False
        if w is not None and w > vw:
            return False
        if h is not None and h > vh:
            return False
        return True

    # 5. Rotation is allowed
    is_upright = False
    if category is not None and category.lower() in UPRIGHT_CARGO_CATEGORIES:
        is_upright = True

    if allow_3d_rotation is True:
        can_3d_rotate = True
    elif allow_3d_rotation is False:
        can_3d_rotate = False
    else:
        can_3d_rotate = not is_upright

    # Case A: All 3 dimensions are present
    if l is not None and w is not None and h is not None:
        if can_3d_rotate:
            for (cl, cw, ch) in itertools.permutations([l, w, h]):
                if cl <= vl and cw <= vw and ch <= vh:
                    return True
            return False
        else:
            # Upright / 2D yaw rotation on bed (H <= vh, and L/W can swap on bed)
            if h > vh:
                return False
            if (l <= vl and w <= vw) or (w <= vl and l <= vw):
                return True
            return False

    # Case B: Partial dimensions
    known_dims = [d for d in [l, w, h] if d is not None]
    if len(known_dims) == 1:
        d = known_dims[0]
        if can_3d_rotate:
            return d <= max(vl, vw, vh)
        else:
            if h is not None:
                return h <= vh
            return d <= max(vl, vw)
    elif len(known_dims) == 2:
        d1, d2 = known_dims
        if can_3d_rotate:
            v_dims = [vl, vw, vh]
            for i in range(3):
                for j in range(3):
                    if i != j and d1 <= v_dims[i] and d2 <= v_dims[j]:
                        return True
            return False
        else:
            if h is not None:
                other = d1 if h == d2 else d2
                if h > vh:
                    return False
                return other <= max(vl, vw)
            return (d1 <= vl and d2 <= vw) or (d2 <= vl and d1 <= vw)

    return True


# Categories identifying motorized / rolling vehicles
VEHICLE_CARGO_CATEGORIES: set = {
    "car",
    "automobile",
    "suv",
    "vehicle",
    "sedan",
    "truck",
    "van",
}


def _is_vehicle_compatible_with_category(vehicle: VehicleSpec, category: Optional[str]) -> bool:
    """
    Validates semantic compatibility between a vehicle specification and a cargo category.
    Specialized vehicles (e.g. car carriers) require matching vehicle cargo.
    """
    if hasattr(vehicle, "is_compatible_with_cargo"):
        return vehicle.is_compatible_with_cargo(category)
    if category is None:
        return True
    if vehicle.vehicle_id == "V_CAR_CARRIER_MULTI" or getattr(vehicle, "vehicle_type", "") == "specialized_car_carrier":
        return category.strip().lower() in VEHICLE_CARGO_CATEGORIES
    return True


def _is_vehicle_compatible_with_shipment(vehicle: VehicleSpec, items: List[Dict[str, Any]]) -> bool:
    """
    Validates semantic compatibility between a vehicle specification and a multi-item shipment.
    Specialized vehicles (e.g. car carriers) require at least one vehicle/car cargo item in the shipment.
    """
    if hasattr(vehicle, "is_compatible_with_shipment"):
        return vehicle.is_compatible_with_shipment(items)
    if not items:
        return True
    if vehicle.vehicle_id == "V_CAR_CARRIER_MULTI" or getattr(vehicle, "vehicle_type", "") == "specialized_car_carrier":
        return any(item.get("category", "").strip().lower() in VEHICLE_CARGO_CATEGORIES for item in items)
    return True


# -----------------------------------------------------------------------------
# Unified Pipeline Class
# -----------------------------------------------------------------------------

class CargoAnalysisPipeline:
    """
    End-to-End Cargo Vision Analysis and Fleet Dispatch Orchestrator.
    Supports:
      - Single Cargo Item Analysis
      - Single-Image Multi-Object Cargo Extraction (`extract_all_objects=True`)
      - Multi-Load Shipment Aggregation (`analyze_multiple`)
      - Multi-Constraint Vehicle Selection (Semantic, Dimensions, Volume, Floor Area, Payload Capacity)
      - Dual vision backends (YOLOv8 COCO detector & MobileNetV2 classification fallback)
    """

    _dimensions_fit_vehicle = staticmethod(_dimensions_fit_vehicle)
    _is_vehicle_compatible_with_category = staticmethod(_is_vehicle_compatible_with_category)
    _is_vehicle_compatible_with_shipment = staticmethod(_is_vehicle_compatible_with_shipment)

    def __init__(
        self,
        model_path: str = "models/cargo_mobilenetv2_expanded_best.keras",
        yolo_model_path: str = "yolov8n.pt",
        custom_yolo_model_path: Optional[str] = "models/cargo_yolo_exp_v2_best.pt",
        custom_vehicle_db: Optional[VehicleDatabase] = None,
        custom_dimension_estimator: Optional[DimensionEstimator] = None,
        detector_backend: str = "auto",
        conf_threshold: float = 0.25,
    ):
        self.model_path = model_path
        self.yolo_model_path = yolo_model_path
        self.custom_yolo_model_path = custom_yolo_model_path
        self.vehicle_db = custom_vehicle_db if custom_vehicle_db is not None else vehicle_db
        self.dimension_estimator = custom_dimension_estimator if custom_dimension_estimator is not None else DimensionEstimator()
        self.detector_backend = detector_backend
        self.conf_threshold = conf_threshold
        self._model = None              # Lazy-loaded MobileNetV2
        self._yolo_model = None         # Lazy-loaded YOLO (COCO)
        self._custom_yolo_model = None  # Lazy-loaded Custom Cargo YOLO

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

    def _get_custom_yolo_model(self):
        """Lazy loads custom trained cargo YOLO model if available."""
        if self._custom_yolo_model is None and self.custom_yolo_model_path is not None:
            if os.path.exists(self.custom_yolo_model_path):
                from ultralytics import YOLO
                self._custom_yolo_model = YOLO(self.custom_yolo_model_path)
            else:
                for fallback_p in ["models/cargo_yolo_exp_v2_best.pt", "models/cargo_yolo_24class_best.pt", "models/cargo_yolo_improved_best.pt"]:
                    if os.path.exists(fallback_p):
                        from ultralytics import YOLO
                        self._custom_yolo_model = YOLO(fallback_p)
                        break
        return self._custom_yolo_model

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
        Executes object detection using specialized custom cargo YOLO model and/or base COCO model.
        Returns detection summary dictionary and warnings, or None if no valid objects found.
        Filters out unsupported / non-cargo categories.
        """
        warnings = []
        try:
            raw_detections = []

            # 1. Custom Cargo YOLO model (specialized for Box and custom cargo categories)
            custom_m = self._get_custom_yolo_model()
            if custom_m is not None:
                try:
                    c_results = custom_m.predict(source=image_input, conf=self.conf_threshold, verbose=False)[0]
                    if c_results.boxes is not None and len(c_results.boxes) > 0:
                        for i in range(len(c_results.boxes)):
                            cls_id = int(c_results.boxes.cls[i].item())
                            raw_cls_name = custom_m.names.get(cls_id, f"class_{cls_id}").lower()
                            c_score = float(c_results.boxes.conf[i].item())
                            xyxy = c_results.boxes.xyxy[i].tolist()
                            if raw_cls_name in COCO_TO_CARGO_MAP:
                                canonical_class = COCO_TO_CARGO_MAP[raw_cls_name]
                                raw_detections.append({
                                    "raw_class": raw_cls_name,
                                    "canonical_class": canonical_class,
                                    "confidence": round(c_score, 4),
                                    "bbox_px": [round(x, 1) for x in xyxy],
                                    "backend": "yolo_cargo_custom",
                                })
                except Exception as e:
                    warnings.append(f"Custom YOLO inference note: {str(e)}")

            # 2. Base COCO YOLO model (provides reliable vehicle/car, refrigerator, etc. detections)
            coco_m = self._get_yolo_model()
            if coco_m is not None:
                try:
                    coco_results = coco_m.predict(source=image_input, conf=self.conf_threshold, verbose=False)[0]
                    if coco_results.boxes is not None and len(coco_results.boxes) > 0:
                        for i in range(len(coco_results.boxes)):
                            cls_id = int(coco_results.boxes.cls[i].item())
                            raw_cls_name = coco_m.names.get(cls_id, f"class_{cls_id}").lower()
                            c_score = float(coco_results.boxes.conf[i].item())
                            xyxy = coco_results.boxes.xyxy[i].tolist()
                            if raw_cls_name in COCO_TO_CARGO_MAP:
                                canonical_class = COCO_TO_CARGO_MAP[raw_cls_name]
                                raw_detections.append({
                                    "raw_class": raw_cls_name,
                                    "canonical_class": canonical_class,
                                    "confidence": round(c_score, 4),
                                    "bbox_px": [round(x, 1) for x in xyxy],
                                    "backend": "yolo_coco",
                                })
                except Exception as e:
                    warnings.append(f"COCO YOLO inference note: {str(e)}")

            if not raw_detections:
                return None, ["No objects detected by YOLO detector above confidence threshold."]

            # Sort candidate detections by descending confidence
            raw_detections.sort(key=lambda d: d["confidence"], reverse=True)

            # Apply Non-Maximum Suppression (IoU overlap deduplication)
            filtered_detections = []
            for d in raw_detections:
                box_a = d["bbox_px"]
                is_duplicate = False
                for existing in filtered_detections:
                    box_b = existing["bbox_px"]
                    xA = max(box_a[0], box_b[0])
                    yA = max(box_a[1], box_b[1])
                    xB = min(box_a[2], box_b[2])
                    yB = min(box_a[3], box_b[3])
                    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
                    boxAArea = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
                    boxBArea = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                    denom = boxAArea + boxBArea - interArea
                    iou = interArea / float(denom) if denom > 0 else 0.0

                    if iou > 0.45:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    filtered_detections.append(d)

            if not filtered_detections:
                return None, ["No supported cargo objects found in YOLO detections."]

            primary = filtered_detections[0]
            primary_instances = [d for d in filtered_detections if d["canonical_class"] == primary["canonical_class"]]
            primary_conf = primary["confidence"]
            primary_reliability = "RELIABLE" if primary_conf >= RELIABLE_CONFIDENCE_THRESHOLD else "REVIEW_REQUIRED"

            if primary_reliability == "REVIEW_REQUIRED":
                warnings.append(
                    f"Detection confidence for '{primary['canonical_class']}' ({primary_conf:.1%}) is below reliable threshold ({RELIABLE_CONFIDENCE_THRESHOLD:.1%}). "
                    "Manual verification recommended before fleet dispatch."
                )

            detection_res = {
                "class_name": primary["canonical_class"],
                "confidence": primary["confidence"],
                "raw_class_name": primary["raw_class"],
                "detector_backend": primary.get("backend", "yolo"),
                "bounding_box_px": primary["bbox_px"],
                "detected_instance_count": len(primary_instances),
                "all_detections": filtered_detections,
                "reliability": primary_reliability,
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
        reliability = "RELIABLE" if confidence >= RELIABLE_CONFIDENCE_THRESHOLD else "REVIEW_REQUIRED"

        probabilities = {
            TARGET_CLASSES[i]: round(float(probs[i]), 4)
            for i in range(len(TARGET_CLASSES))
        }

        if reliability == "REVIEW_REQUIRED":
            warnings.append(
                f"Classification confidence for '{pred_class}' ({confidence:.1%}) is below reliable threshold ({RELIABLE_CONFIDENCE_THRESHOLD:.1%}). "
                "Image may be ambiguous, poorly lit, or out-of-distribution; manual review recommended."
            )

        classification_res = {
            "class_name": pred_class,
            "confidence": round(confidence, 4),
            "detector_backend": "mobilenetv2",
            "reliability": reliability,
            "probabilities": probabilities,
        }

        return classification_res, warnings

    def calculate_cargo_requirements(
        self,
        dimensions: Dict[str, Optional[float]],
        category: str,
        quantity: int,
        unit_weight_kg: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Calculates unit volume, required total volume, stacking floor area, and payload weight.
        """
        warnings = []
        length = dimensions.get("length_cm")
        width = dimensions.get("width_cm")
        height = dimensions.get("height_cm")

        # Resolve weight from argument, dimensions dict, or category prior
        if unit_weight_kg is None:
            unit_weight_kg = dimensions.get("weight_kg")

        stacking_limit = CATEGORY_STACKING_LIMITS.get(category.lower(), 1)
        packing_factor = CATEGORY_PACKING_FACTORS.get(category.lower(), 0.70)

        unit_vol_m3 = None
        total_vol_m3 = None
        req_floor_area_m2 = None
        total_weight_kg = None

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

        if unit_weight_kg is not None:
            total_weight_kg = round(unit_weight_kg * quantity, 2)

        cargo_summary = {
            "total_items": quantity,
            "unit_volume_m3": unit_vol_m3,
            "total_volume_m3": total_vol_m3,
            "unit_weight_kg": unit_weight_kg,
            "total_weight_kg": total_weight_kg,
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
        Matches single-category cargo requirements against the structured Vehicle Database using hard constraints:
          1. Physical dimensions fit (length, width, height)
          2. Usable volume fit
          3. Floor bed area fit
          4. Payload weight fit
        """
        warnings = []
        total_weight_kg = cargo_summary.get("total_weight_kg")
        if total_weight_kg is not None:
            warnings.append(
                f"Payload suitability cannot be verified against a physical scale; weight ({total_weight_kg:.1f} kg) is estimated from standard category priors."
            )
        else:
            warnings.append(
                "Payload suitability cannot be verified because physical object weight is unavailable."
            )

        length = dimensions.get("length_cm")
        width = dimensions.get("width_cm")
        height = dimensions.get("height_cm")
        total_vol = cargo_summary.get("total_volume_m3")
        req_floor_area = cargo_summary.get("required_floor_area_m2")

        all_vehicles = self.vehicle_db.get_all_vehicles()
        suitable_vehicles: List[VehicleSpec] = []

        for v in all_vehicles:
            # Constraint 0: Semantic compatibility fit
            if not _is_vehicle_compatible_with_category(v, category):
                continue

            # Constraint 1: Single item physical dimension fit (with valid rotation support)
            if not _dimensions_fit_vehicle(dimensions, v, allow_rotation=True, category=category):
                continue

            # Constraint 2: Total required volume fit
            if total_vol is not None and total_vol > v.usable_volume_m3:
                continue

            # Constraint 3: Required floor area fit
            if req_floor_area is not None and req_floor_area > v.floor_area_m2:
                continue

            # Constraint 4: Payload weight fit
            if total_weight_kg is not None and total_weight_kg > v.max_payload_kg:
                continue

            suitable_vehicles.append(v)

        if not suitable_vehicles:
            # Fallback: cargo exceeds single standard vehicle capacity
            compatible_vehicles = [v for v in all_vehicles if _is_vehicle_compatible_with_category(v, category)]
            largest_v = compatible_vehicles[-1] if compatible_vehicles else all_vehicles[-1]
            exceeded_reasons = []
            if not _dimensions_fit_vehicle(dimensions, largest_v, allow_rotation=True, category=category):
                exceeded_reasons.append("physical dimensions")
            if total_vol is not None and total_vol > largest_v.usable_volume_m3:
                exceeded_reasons.append("volume")
            if req_floor_area is not None and req_floor_area > largest_v.floor_area_m2:
                exceeded_reasons.append("floor bed area")
            if total_weight_kg is not None and total_weight_kg > largest_v.max_payload_kg:
                exceeded_reasons.append("payload weight capacity")

            exceeded_details = f" ({', '.join(exceeded_reasons)})" if exceeded_reasons else ""

            return {
                "vehicle_id": largest_v.vehicle_id,
                "vehicle_name": largest_v.vehicle_name,
                "reason": (
                    f"Cargo requirement ({quantity} {category}s) exceeds standard single-vehicle capacity{exceeded_details}. "
                    f"Multi-trip or fleet batching using {largest_v.vehicle_name} is required."
                ),
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
        if total_weight_kg is not None:
            reason_parts.append(f"with {total_weight_kg:.1f} kg estimated payload (vehicle payload limit: {primary_v.max_payload_kg:.1f} kg)")
        
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
        total_weight_kg: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Matches multi-item combined cargo shipment requirements against the Vehicle Database:
          1. Individual dimension fit (length, width, height with valid rotation)
          2. Total usable volume fit
          3. Total floor bed area fit
          4. Total payload weight fit
        """
        warnings = []
        if total_weight_kg is not None:
            warnings.append(
                f"Payload suitability cannot be verified against a physical scale; total shipment weight ({total_weight_kg:.1f} kg) is estimated from category priors."
            )
        else:
            warnings.append(
                "Payload suitability cannot be verified because physical object weight is unavailable."
            )

        if not items:
            return {
                "vehicle_id": None,
                "vehicle_name": None,
                "reason": "Shipment is empty. No cargo items to transport.",
                "alternatives": [],
            }, ["No cargo items provided for vehicle recommendation."]

        all_vehicles = self.vehicle_db.get_all_vehicles()
        suitable_vehicles: List[VehicleSpec] = []

        for v in all_vehicles:
            # Constraint 0: Semantic compatibility fit
            if not _is_vehicle_compatible_with_shipment(v, items):
                continue

            # Constraint 1: Every individual item's dimensions must fit inside vehicle (with valid rotation)
            fits_dimensions = True
            for item in items:
                dims = item.get("dimensions", {})
                cat = item.get("category")
                if not _dimensions_fit_vehicle(dims, v, allow_rotation=True, category=cat):
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

            # Constraint 4: Total payload weight fit
            if total_weight_kg is not None and total_weight_kg > v.max_payload_kg:
                continue

            suitable_vehicles.append(v)

        total_quantity = sum(item.get("quantity", 1) for item in items)
        category_summary_str = ", ".join(f"{item.get('quantity', 1)} {item.get('category', 'item')}(s)" for item in items)

        if not suitable_vehicles:
            compatible_vehicles = [v for v in all_vehicles if _is_vehicle_compatible_with_shipment(v, items)]
            largest_v = compatible_vehicles[-1] if compatible_vehicles else all_vehicles[-1]
            exceeded_reasons = []
            dim_exceeded = any(
                not _dimensions_fit_vehicle(item.get("dimensions", {}), largest_v, allow_rotation=True, category=item.get("category"))
                for item in items
            )
            if dim_exceeded:
                exceeded_reasons.append("individual item dimensions")
            if total_volume_m3 is not None and total_volume_m3 > largest_v.usable_volume_m3:
                exceeded_reasons.append("volume")
            if total_floor_area_m2 is not None and total_floor_area_m2 > largest_v.floor_area_m2:
                exceeded_reasons.append("floor bed area")
            if total_weight_kg is not None and total_weight_kg > largest_v.max_payload_kg:
                exceeded_reasons.append("payload weight capacity")

            exceeded_details = f" ({', '.join(exceeded_reasons)})" if exceeded_reasons else ""

            return {
                "vehicle_id": largest_v.vehicle_id,
                "vehicle_name": largest_v.vehicle_name,
                "reason": (
                    f"Combined shipment ({total_quantity} items: {category_summary_str}) exceeds "
                    f"standard single-vehicle capacity{exceeded_details}. Multi-trip or fleet batching using {largest_v.vehicle_name} is required."
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
        if total_weight_kg is not None:
            reason_parts.append(f"with {total_weight_kg:.1f} kg estimated payload (vehicle payload limit: {primary_v.max_payload_kg:.1f} kg)")

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
        extract_all_objects: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end analysis on a single cargo photo.

        Args:
            image_path: Path to the image file.
            quantity: Quantity multiplier for single-item mode (default: 1).
            known_marker_size_cm: Optional ArUco marker size for Mode 1 measurement.
            object_bbox_px: Optional bounding box override.
            extract_all_objects: When True, extracts ALL valid detected cargo objects in the image
                                and aggregates their physical requirements into a combined shipment.
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
        yolo_res = None

        if self.detector_backend in ("auto", "yolo"):
            yolo_res, yolo_warnings = self.detect_with_yolo(image_path)
            if yolo_res is not None:
                all_warnings.extend(yolo_warnings)
                classification_res = yolo_res
                detected_category = yolo_res["class_name"]
                if object_bbox_px is None and "bounding_box_px" in yolo_res:
                    auto_bbox = tuple(int(round(coord)) for coord in yolo_res["bounding_box_px"])

        # 3b. In-Image Multi-Object Extraction Workflow
        if extract_all_objects and yolo_res is not None and len(yolo_res.get("all_detections", [])) > 0:
            all_dets = yolo_res["all_detections"]

            # Group detections by canonical class
            category_counts: Dict[str, int] = {}
            category_boxes: Dict[str, List[List[float]]] = {}
            category_confs: Dict[str, List[float]] = {}

            for d in all_dets:
                cat = d["canonical_class"]
                category_counts[cat] = category_counts.get(cat, 0) + 1
                category_boxes.setdefault(cat, []).append(d["bbox_px"])
                category_confs.setdefault(cat, []).append(d["confidence"])

            multi_items = []
            review_required_items = []
            for cat, count in category_counts.items():
                qty = count * (quantity if quantity > 1 else 1)
                dim_res = self.dimension_estimator.estimate(category=cat, fallback_to_prior=True)
                all_warnings.extend(dim_res.warnings)

                dims_dict = {
                    "status": dim_res.status,
                    "source": dim_res.source,
                    "length_cm": dim_res.dimensions_cm.get("length"),
                    "width_cm": dim_res.dimensions_cm.get("width"),
                    "height_cm": dim_res.dimensions_cm.get("height"),
                    "weight_kg": getattr(dim_res, "weight_kg", None),
                    "confidence": dim_res.confidence,
                }

                c_sum, c_warn = self.calculate_cargo_requirements(
                    dimensions=dims_dict,
                    category=cat,
                    quantity=qty,
                    unit_weight_kg=dims_dict.get("weight_kg"),
                )
                all_warnings.extend(c_warn)

                avg_conf = round(sum(category_confs[cat]) / len(category_confs[cat]), 4)
                item_reliability = "RELIABLE" if avg_conf >= RELIABLE_CONFIDENCE_THRESHOLD else "REVIEW_REQUIRED"
                is_dispatch_ready = (item_reliability == "RELIABLE")

                if item_reliability == "REVIEW_REQUIRED":
                    review_required_items.append(cat)
                    all_warnings.append(
                        f"Low-confidence cargo detected: '{cat}' ({avg_conf:.1%}) is marked REVIEW_REQUIRED. "
                        "Verify presence before finalizing vehicle dispatch."
                    )

                multi_items.append({
                    "category": cat,
                    "quantity": qty,
                    "confidence": avg_conf,
                    "reliability": item_reliability,
                    "is_dispatch_ready": is_dispatch_ready,
                    "detected_instances": count,
                    "bounding_boxes_px": category_boxes[cat],
                    "dimensions": dims_dict,
                    "cargo_summary": c_sum,
                })

            total_items = sum(item["quantity"] for item in multi_items)
            volumes = [item["cargo_summary"]["total_volume_m3"] for item in multi_items]
            floor_areas = [item["cargo_summary"]["required_floor_area_m2"] for item in multi_items]
            weights = [item["cargo_summary"]["total_weight_kg"] for item in multi_items]

            total_volume_m3 = round(sum(v for v in volumes if v is not None), 4) if all(v is not None for v in volumes) else None
            total_floor_area_m2 = round(sum(a for a in floor_areas if a is not None), 4) if all(a is not None for a in floor_areas) else None
            total_weight_kg = round(sum(w for w in weights if w is not None), 2) if all(w is not None for w in weights) else None

            reliable_count = sum(item["quantity"] for item in multi_items if item["reliability"] == "RELIABLE")
            review_count = sum(item["quantity"] for item in multi_items if item["reliability"] == "REVIEW_REQUIRED")

            shipment_summary = {
                "total_items": total_items,
                "total_volume_m3": total_volume_m3,
                "total_floor_area_m2": total_floor_area_m2,
                "total_weight_kg": total_weight_kg,
                "reliability_breakdown": {
                    "reliable_items_count": reliable_count,
                    "review_required_items_count": review_count,
                    "review_required_categories": review_required_items,
                },
                "item_breakdown": [
                    {
                        "category": item["category"],
                        "quantity": item["quantity"],
                        "confidence": item["confidence"],
                        "reliability": item["reliability"],
                        "is_dispatch_ready": item["is_dispatch_ready"],
                        "unit_volume_m3": item["cargo_summary"]["unit_volume_m3"],
                        "total_volume_m3": item["cargo_summary"]["total_volume_m3"],
                        "unit_weight_kg": item["cargo_summary"]["unit_weight_kg"],
                        "total_weight_kg": item["cargo_summary"]["total_weight_kg"],
                        "required_floor_area_m2": item["cargo_summary"]["required_floor_area_m2"],
                        "dimensions_cm": {
                            "length": item["dimensions"]["length_cm"],
                            "width": item["dimensions"]["width_cm"],
                            "height": item["dimensions"]["height_cm"],
                        },
                    }
                    for item in multi_items
                ],
            }

            vehicle_rec, veh_warnings = self.recommend_vehicle_for_shipment(
                items=multi_items,
                total_volume_m3=total_volume_m3,
                total_floor_area_m2=total_floor_area_m2,
                total_weight_kg=total_weight_kg,
            )
            all_warnings.extend(veh_warnings)

            # Deduplicate warnings while preserving order
            seen_w = set()
            deduped_warnings = []
            for w in all_warnings:
                if w not in seen_w:
                    seen_w.add(w)
                    deduped_warnings.append(w)

            return {
                "status": "SUCCESS",
                "mode": "SINGLE_IMAGE_MULTI_OBJECT",
                "input_image": os.path.abspath(image_path),
                "detected_objects": multi_items,
                "total_items": total_items,
                "shipment_summary": shipment_summary,
                "vehicle_recommendation": vehicle_rec,
                "warnings": deduped_warnings,
            }

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

        # 4. Dimension & Weight Estimation
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
            "weight_kg": getattr(dim_res, "weight_kg", None),
            "confidence": dim_res.confidence,
        }

        # 5. Total Cargo Calculation
        cargo_summary, cargo_warnings = self.calculate_cargo_requirements(
            dimensions=dimensions_dict,
            category=detected_category,
            quantity=quantity,
            unit_weight_kg=dimensions_dict.get("weight_kg"),
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
        rel_status = classification_res.get(
            "reliability",
            "RELIABLE" if classification_res.get("confidence", 0.0) >= RELIABLE_CONFIDENCE_THRESHOLD else "REVIEW_REQUIRED"
        )
        return {
            "status": "SUCCESS",
            "mode": "SINGLE_LOAD",
            "input_image": os.path.abspath(image_path),
            "classification": classification_res,
            "reliability": rel_status,
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
                "reliability": single_res.get("reliability", single_res["classification"].get("reliability", "RELIABLE")),
                "quantity": qty,
                "dimensions": single_res["dimensions"],
                "cargo_summary": single_res["cargo_summary"],
            }
            shipment_items.append(item_entry)

        # Volumetric, Weight & Space Aggregation
        total_items = sum(item["quantity"] for item in shipment_items)
        volumes = [item["cargo_summary"]["total_volume_m3"] for item in shipment_items]
        floor_areas = [item["cargo_summary"]["required_floor_area_m2"] for item in shipment_items]
        weights = [item["cargo_summary"]["total_weight_kg"] for item in shipment_items]

        total_volume_m3 = round(sum(v for v in volumes if v is not None), 4) if all(v is not None for v in volumes) else None
        total_floor_area_m2 = round(sum(a for a in floor_areas if a is not None), 4) if all(a is not None for a in floor_areas) else None
        total_weight_kg = round(sum(w for w in weights if w is not None), 2) if all(w is not None for w in weights) else None

        shipment_summary = {
            "total_items": total_items,
            "total_volume_m3": total_volume_m3,
            "total_floor_area_m2": total_floor_area_m2,
            "total_weight_kg": total_weight_kg,
            "item_breakdown": [
                {
                    "category": item["category"],
                    "quantity": item["quantity"],
                    "unit_volume_m3": item["cargo_summary"]["unit_volume_m3"],
                    "total_volume_m3": item["cargo_summary"]["total_volume_m3"],
                    "unit_weight_kg": item["cargo_summary"]["unit_weight_kg"],
                    "total_weight_kg": item["cargo_summary"]["total_weight_kg"],
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
            total_weight_kg=total_weight_kg,
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
    parser.add_argument("--extract-all", action="store_true", help="Extract and aggregate all detected cargo objects in a single image.")
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
            dim_str = f"{dims['length_cm']} x {dims['width_cm']} x {dims['height_cm']} cm" if dims.get('length_cm') else "Unmeasured"
            vol_str = f"{c_sum['total_volume_m3']:.4f} m³" if c_sum.get('total_volume_m3') else "N/A"
            wt_str = f"{c_sum['total_weight_kg']:.1f} kg" if c_sum.get('total_weight_kg') is not None else "N/A"
            print(f"{idx+1:>2}. {cat:<16} x{qty:<3} (Unit: {dim_str}, Vol: {vol_str}, Wt: {wt_str})")

        print(f"\nTOTAL ITEMS:      {result['total_items']}")
        print(f"TOTAL VOLUME:     {summary['total_volume_m3']:.4f} m³" if summary.get('total_volume_m3') else "TOTAL VOLUME:     N/A")
        print(f"TOTAL FLOOR AREA: {summary['total_floor_area_m2']:.4f} m²" if summary.get('total_floor_area_m2') else "TOTAL FLOOR AREA: N/A")
        print(f"TOTAL WEIGHT:     {summary['total_weight_kg']:.2f} kg (estimated)" if summary.get('total_weight_kg') is not None else "TOTAL WEIGHT:     N/A")
        print("-" * 78)
        print("RECOMMENDED VEHICLE")
        print("-" * 78)
        print(f"Vehicle:          {rec['vehicle_name']}")
        print(f"Vehicle ID:       {rec['vehicle_id']}")
        reason_text = textwrap.fill(rec['reason'], width=76, initial_indent="Reason:           ", subsequent_indent="                  ")
        print(reason_text)
        if rec.get("alternatives"):
            alts_str = ", ".join(rec["alternatives"])
            alts_text = textwrap.fill(alts_str, width=76, initial_indent="Alternatives:     ", subsequent_indent="                  ")
            print(alts_text)
        print("-" * 78)
        print("IMPORTANT NOTICES & DISCLAIMERS:")
        for w in result["warnings"]:
            w_text = textwrap.fill(f"* {w}", width=76, initial_indent="  ", subsequent_indent="    ")
            print(w_text)
        print("=" * 78)
        return

    # Single-Image Mode (Single Object or In-Image Multi-Object)
    result = pipeline.analyze(
        image_path=args.image,
        quantity=args.quantity,
        known_marker_size_cm=args.marker_size,
        object_bbox_px=tuple(args.bbox) if args.bbox else None,
        extract_all_objects=args.extract_all,
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

    # Single-Image Multi-Object Report
    if result.get("mode") == "SINGLE_IMAGE_MULTI_OBJECT":
        rec = result["vehicle_recommendation"]
        summary = result["shipment_summary"]

        print("\n" + "=" * 78)
        print("CARGO VISION - SINGLE-IMAGE MULTI-OBJECT DISPATCH REPORT")
        print("=" * 78)
        print(f"Input Image:           {result['input_image']}")
        print("\nDETECTED OBJECTS")
        print("-" * 78)
        for idx, item in enumerate(result["detected_objects"]):
            cat = item["category"].capitalize()
            qty = item["quantity"]
            dims = item["dimensions"]
            c_sum = item["cargo_summary"]
            conf = item["confidence"]
            rel = item.get("reliability", "RELIABLE")
            rel_tag = f" [{rel}]" if rel == "REVIEW_REQUIRED" else ""
            dim_str = f"{dims['length_cm']} x {dims['width_cm']} x {dims['height_cm']} cm" if dims.get('length_cm') else "Unmeasured"
            vol_str = f"{c_sum['total_volume_m3']:.4f} m³" if c_sum.get('total_volume_m3') else "N/A"
            wt_str = f"{c_sum['total_weight_kg']:.1f} kg" if c_sum.get('total_weight_kg') is not None else "N/A"
            print(f"{idx+1:>2}. {cat:<16} x{qty:<3} (Conf: {conf:.1%}{rel_tag}, Unit: {dim_str}, Vol: {vol_str}, Wt: {wt_str})")

        print(f"\nTOTAL ITEMS:      {result['total_items']}")
        print(f"TOTAL VOLUME:     {summary['total_volume_m3']:.4f} m³" if summary.get('total_volume_m3') else "TOTAL VOLUME:     N/A")
        print(f"TOTAL FLOOR AREA: {summary['total_floor_area_m2']:.4f} m²" if summary.get('total_floor_area_m2') else "TOTAL FLOOR AREA: N/A")
        print(f"TOTAL WEIGHT:     {summary['total_weight_kg']:.2f} kg (estimated)" if summary.get('total_weight_kg') is not None else "TOTAL WEIGHT:     N/A")
        print("-" * 78)
        print("RECOMMENDED VEHICLE")
        print("-" * 78)
        print(f"Vehicle:          {rec['vehicle_name']}")
        print(f"Vehicle ID:       {rec['vehicle_id']}")
        reason_text = textwrap.fill(rec['reason'], width=76, initial_indent="Reason:           ", subsequent_indent="                  ")
        print(reason_text)
        if rec.get("alternatives"):
            alts_str = ", ".join(rec["alternatives"])
            alts_text = textwrap.fill(alts_str, width=76, initial_indent="Alternatives:     ", subsequent_indent="                  ")
            print(alts_text)
        print("-" * 78)
        print("IMPORTANT NOTICES & DISCLAIMERS:")
        for w in result["warnings"]:
            w_text = textwrap.fill(f"* {w}", width=76, initial_indent="  ", subsequent_indent="    ")
            print(w_text)
        print("=" * 78)
        return

    # Single-Load Report (Standard Single Item Mode)
    cls_info = result["classification"]
    dim_info = result["dimensions"]
    summary = result["cargo_summary"]
    rec = result["vehicle_recommendation"]
    rel_status = result.get("reliability", cls_info.get("reliability", "RELIABLE"))
    rel_tag = f" [{rel_status}]" if rel_status == "REVIEW_REQUIRED" else ""

    print("\n" + "=" * 78)
    print("CARGO VISION LOGISTICS DISPATCH REPORT")
    print("=" * 78)
    print(f"Input Image:           {result['input_image']}")
    print(f"Detected Cargo:        {cls_info['class_name'].upper()} (Confidence: {cls_info['confidence']:.1%}{rel_tag})")
    if "bounding_box_px" in cls_info:
        print(f"Bounding Box:          {cls_info['bounding_box_px']} px")
    print(f"Quantity:              {result['quantity']} item(s)")
    print("-" * 78)
    print("PHYSICAL DIMENSION ESTIMATE:")
    print(f"  - Status:            {dim_info['status']} ({dim_info['source']})")
    print(f"  - Length:            {dim_info['length_cm']:.1f} cm" if dim_info.get('length_cm') is not None else "  - Length:            None")
    print(f"  - Width (Depth):     {dim_info['width_cm']:.1f} cm" if dim_info.get('width_cm') is not None else "  - Width (Depth):     None")
    print(f"  - Height:            {dim_info['height_cm']:.1f} cm" if dim_info.get('height_cm') is not None else "  - Height:            None")
    print(f"  - Dimension Conf:    {dim_info['confidence']:.1%}")
    print("-" * 78)
    print("CARGO VOLUME & SPACE REQUIREMENTS:")
    print(f"  - Unit Volume:       {summary['unit_volume_m3']} m³" if summary.get('unit_volume_m3') is not None else "  - Unit Volume:       None")
    print(f"  - Total Req Volume:  {summary['total_volume_m3']} m³ (adjusted for packing)" if summary.get('total_volume_m3') is not None else "  - Total Req Volume:  None")
    print(f"  - Floor Area Bed:    {summary['required_floor_area_m2']} m² (max {summary['stacking_limit_layers']} stack layers)" if summary.get('required_floor_area_m2') is not None else "  - Floor Area Bed:    None")
    if summary.get("unit_weight_kg") is not None:
        print(f"  - Unit Weight:       {summary['unit_weight_kg']:.1f} kg (estimated category prior)")
        print(f"  - Total Weight:      {summary['total_weight_kg']:.1f} kg")
    print("-" * 78)
    print("FLEET RECOMMENDATION:")
    print(f"  - Recommended:       {rec['vehicle_name']}")
    print(f"  - Vehicle ID:        {rec['vehicle_id']}")
    reason_text = textwrap.fill(rec['reason'], width=76, initial_indent="  - Reason:            ", subsequent_indent="                       ")
    print(reason_text)
    if rec.get("alternatives"):
        alts_str = ", ".join(rec["alternatives"])
        alts_text = textwrap.fill(alts_str, width=76, initial_indent="  - Alternatives:      ", subsequent_indent="                       ")
        print(alts_text)
    print("-" * 78)
    print("IMPORTANT NOTICES & DISCLAIMERS:")
    for w in result["warnings"]:
        w_text = textwrap.fill(f"* {w}", width=76, initial_indent="  ", subsequent_indent="    ")
        print(w_text)
    print("=" * 78)



if __name__ == "__main__":
    main()