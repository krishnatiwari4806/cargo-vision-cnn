"""
dimension_estimator.py - Core Physical Dimension Estimation Engine
===================================================================
Project: Cargo Vision Logistics System (BUG #6 Upgrade V1)
Purpose: Provides an explicit, multi-tier physical dimension estimation engine:
         1. MODE 1 - Reference-Marker Measurement (High-precision fiducial scale detection via OpenCV).
         2. MODE 1b - Reference Scale + 3D Aspect-Ratio Depth Inference (ARUCO_PLUS_ASPECT_PRIOR).
         3. MODE 2 - Category Prior Fallback (Configurable industry-standard parametric dimension envelopes).

Physical & Geometric Reality Notice:
-------------------------------------
1. Monocular Scale Ambiguity:
   In standard monocular RGB vision, physical 3D coordinates (X, Y, Z) project to 2D pixel coordinates (x, y)
   via the pinhole projection equation: x = f * (X / Z).
   Without knowing the camera-to-object distance Z, camera focal length f, or a metric scale anchor,
   it is mathematically impossible to measure metric centimeters from a 2D image alone.

2. Planar Reference Approximation & Limitations:
   Calculating physical dimensions as (pixel_width * scale_cm_per_px) assumes the object lies on the same
   depth plane Z as the reference marker. If the object extends significantly in depth (perspective foreshortening)
   or the camera view is oblique, 2D bounding boxes represent projected 2D extents rather than true 3D bounding boxes.
   Furthermore, a single frontal 2D view cannot directly measure the 3rd orthogonal depth dimension without
   multi-view geometry, calibrated depth sensors, or category aspect-ratio inference.

3. Accuracy Disclaimer:
   This module implements geometric and calibration algorithms for dimension estimation.
   It does NOT claim 95% production accuracy until validated against a physically measured ground-truth benchmark.
"""

import os
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import cv2
from PIL import Image


# -----------------------------------------------------------------------------
# Enumerations & Result Schemas
# -----------------------------------------------------------------------------

class EstimationStatus(str, Enum):
    MEASURED = "MEASURED"
    REFERENCE_REQUIRED = "REFERENCE_REQUIRED"
    PRIOR_ESTIMATE = "PRIOR_ESTIMATE"
    ERROR = "ERROR"


class MeasurementSource(str, Enum):
    ARUCO_REFERENCE = "aruco_reference"
    CATEGORY_PRIOR = "category_prior"
    ARUCO_PLUS_ASPECT_PRIOR = "aruco_plus_aspect_prior"
    NONE = "none"


class EstimationMethod(str, Enum):
    ARUCO_REFERENCE = "ARUCO_REFERENCE"
    CATEGORY_PRIOR = "CATEGORY_PRIOR"
    ARUCO_PLUS_ASPECT_PRIOR = "ARUCO_PLUS_ASPECT_PRIOR"
    NONE = "NONE"


@dataclass
class DimensionEstimateResult:
    """
    Standardized, strongly-typed result schema for physical dimension estimation.
    Maintains 100% backward compatibility with existing callers and adds explicit
    estimation metadata, uncertainty bounds, and depth method tracking.
    """
    status: str
    source: Optional[str]
    dimensions_cm: Dict[str, Optional[float]]
    confidence: float
    scale_cm_per_pixel: Optional[float] = None
    weight_kg: Optional[float] = None
    reference: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    estimation_method: Optional[str] = None
    uncertainty_cm: Dict[str, Optional[float]] = field(default_factory=dict)
    uncertainty_percent: Dict[str, Optional[float]] = field(default_factory=dict)
    measurement_confidence: Optional[float] = None
    depth_estimation_method: Optional[str] = None
    weight_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the result to a standard dictionary."""
        return asdict(self)


# -----------------------------------------------------------------------------
# Mode 2: Configurable Category Dimension Priors
# -----------------------------------------------------------------------------

DEFAULT_CATEGORY_PRIORS: Dict[str, Dict[str, Any]] = {
    "box": {
        "length_cm": 45.0,
        "width_cm": 35.0,
        "height_cm": 30.0,
        "description": "Standard medium shipping carton / corrugated box",
        "default_weight_kg": 12.0,
        "confidence": 0.65,
    },
    "chair": {
        "length_cm": 60.0,
        "width_cm": 60.0,
        "height_cm": 90.0,
        "description": "Standard office / dining chair",
        "default_weight_kg": 8.5,
        "confidence": 0.60,
    },
    "couch": {
        "length_cm": 210.0,
        "width_cm": 90.0,
        "height_cm": 85.0,
        "description": "Standard 3-seater living room sofa",
        "default_weight_kg": 65.0,
        "confidence": 0.70,
    },
    "table": {
        "length_cm": 120.0,
        "width_cm": 75.0,
        "height_cm": 75.0,
        "description": "Standard study / dining desk table",
        "default_weight_kg": 25.0,
        "confidence": 0.65,
    },
    "suitcase": {
        "length_cm": 55.0,
        "width_cm": 38.0,
        "height_cm": 23.0,
        "description": "Standard medium check-in / rolling luggage",
        "default_weight_kg": 18.0,
        "confidence": 0.70,
    },
    "car": {
        "length_cm": 450.0,
        "width_cm": 180.0,
        "height_cm": 145.0,
        "description": "Standard mid-size 5-passenger passenger sedan",
        "default_weight_kg": 1400.0,
        "confidence": 0.75,
    },
    "refrigerator": {
        "length_cm": 70.0,
        "width_cm": 70.0,
        "height_cm": 175.0,
        "description": "Standard domestic single/double door refrigerator",
        "default_weight_kg": 75.0,
        "confidence": 0.70,
    },
    "tv": {
        "length_cm": 120.0,
        "width_cm": 15.0,
        "height_cm": 75.0,
        "description": "Standard 50-inch flat panel television",
        "default_weight_kg": 15.0,
        "confidence": 0.70,
    },
    "bed": {
        "length_cm": 200.0,
        "width_cm": 160.0,
        "height_cm": 60.0,
        "description": "Standard double / queen bed frame and mattress",
        "default_weight_kg": 50.0,
        "confidence": 0.65,
    },
    "desk": {
        "length_cm": 120.0,
        "width_cm": 60.0,
        "height_cm": 75.0,
        "description": "Standard office computer / workstation desk",
        "default_weight_kg": 25.0,
        "confidence": 0.65,
    },
}


class CategoryPriorEstimator:
    """
    Mode 2: Resolves physical dimensions using category-specific parametric envelopes.
    Explicitly labels all output as prior estimates, never as measured dimensions.
    """

    def __init__(self, priors: Optional[Dict[str, Dict[str, Any]]] = None):
        self._priors = priors if priors is not None else dict(DEFAULT_CATEGORY_PRIORS)

    def get_supported_categories(self) -> List[str]:
        """Returns list of categories configured with dimension priors."""
        return list(self._priors.keys())

    def estimate(
        self,
        category: str,
        detected_aspect_ratio_wh: Optional[float] = None,
    ) -> DimensionEstimateResult:
        """
        Estimates dimensions from category priors.
        
        Args:
            category: Cargo class name (e.g., 'box', 'couch', 'car').
            detected_aspect_ratio_wh: Optional 2D bbox width/height ratio for aspect adjustment.
        """
        cat_key = category.strip().lower()
        if cat_key not in self._priors:
            return DimensionEstimateResult(
                status=EstimationStatus.ERROR.value,
                source=None,
                estimation_method=EstimationMethod.NONE.value,
                dimensions_cm={"length": None, "width": None, "height": None},
                confidence=0.0,
                measurement_confidence=0.0,
                scale_cm_per_pixel=None,
                reference={"type": None, "size_cm": None, "detected": False},
                warnings=[f"Unknown category '{category}'. No prior dimensions configured."],
            )

        prior = self._priors[cat_key]
        length = float(prior["length_cm"])
        width = float(prior["width_cm"])
        height = float(prior["height_cm"])
        conf = float(prior.get("confidence", 0.60))
        weight = float(prior["default_weight_kg"]) if "default_weight_kg" in prior and prior["default_weight_kg"] is not None else None

        warnings = [
            f"Dimensions for '{cat_key}' are estimated from standard category priors ({prior.get('description', '')}).",
            "This is NOT a direct physical measurement. Actual dimensions may vary.",
            "User verification or physical tape measurement is recommended before fleet dispatch.",
        ]

        # Deterministic population variability uncertainty (+/- 25% for unconstrained category envelope)
        uncertainty_percent = {"length": 25.0, "width": 25.0, "height": 25.0}
        uncertainty_cm = {
            "length": round(length * 0.25, 1),
            "width": round(width * 0.25, 1),
            "height": round(height * 0.25, 1),
        }

        return DimensionEstimateResult(
            status=EstimationStatus.PRIOR_ESTIMATE.value,
            source=MeasurementSource.CATEGORY_PRIOR.value,
            estimation_method=EstimationMethod.CATEGORY_PRIOR.value,
            dimensions_cm={
                "length": round(length, 1),
                "width": round(width, 1),
                "height": round(height, 1),
            },
            confidence=conf,
            measurement_confidence=conf,
            scale_cm_per_pixel=None,
            weight_kg=round(weight, 2) if weight is not None else None,
            weight_source="category_prior" if weight is not None else None,
            depth_estimation_method="category_prior_default",
            uncertainty_percent=uncertainty_percent,
            uncertainty_cm=uncertainty_cm,
            reference={"type": "category_prior", "size_cm": None, "detected": False},
            warnings=warnings,
        )


# -----------------------------------------------------------------------------
# Mode 1: Reference-Marker Measurement Pipeline
# -----------------------------------------------------------------------------

class ReferenceMarkerEstimator:
    """
    Mode 1: Measures physical dimensions using a detected ArUco fiducial marker of known size.
    Calculates precise metric scale (cm/pixel) on the reference plane.
    Supports conservative 3D aspect-ratio depth inference (Mode 1b) when category prior is available.
    """

    def __init__(self, aruco_dict_type: int = cv2.aruco.DICT_4X4_50):
        self.aruco_dict_type = aruco_dict_type
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, detector_params)
        else:
            self.detector = None

    def _load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Loads image as a standard BGR numpy array."""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image path not found: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"OpenCV failed to decode image: {image_input}")
            return img
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def detect_marker(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
    ) -> Tuple[bool, Optional[np.ndarray], Optional[int]]:
        """
        Detects ArUco marker in the image.
        
        Returns:
            Tuple of (detected: bool, corners: np.ndarray, marker_id: int)
        """
        img = self._load_image(image_input)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)

        if ids is not None and len(ids) > 0:
            # Select first detected marker (safely extract scalar ID across OpenCV versions)
            marker_id = int(np.array(ids).flatten()[0])
            return True, corners[0][0], marker_id
        return False, None, None

    def calculate_scale_cm_per_pixel(
        self,
        marker_corners: np.ndarray,
        known_marker_size_cm: float,
    ) -> Tuple[float, List[str]]:
        """
        Calculates metric scale (cm/px) from marker corner points.
        
        Args:
            marker_corners: 4x2 array of corner coordinates [[x0, y0], [x1, y1], [x2, y2], [x3, y3]].
            known_marker_size_cm: Physical side length of square marker in cm.
        
        Returns:
            Tuple of (scale_cm_per_pixel: float, warnings: List[str])
        """
        if known_marker_size_cm <= 0:
            raise ValueError(f"known_marker_size_cm must be positive, got {known_marker_size_cm}")

        p0, p1, p2, p3 = marker_corners

        # 4 edge lengths in pixels
        top_px = float(np.linalg.norm(p1 - p0))
        right_px = float(np.linalg.norm(p2 - p1))
        bottom_px = float(np.linalg.norm(p3 - p2))
        left_px = float(np.linalg.norm(p0 - p3))

        avg_pixel_width = (top_px + bottom_px) / 2.0
        avg_pixel_height = (left_px + right_px) / 2.0
        avg_side_px = (avg_pixel_width + avg_pixel_height) / 2.0

        if avg_side_px <= 0:
            raise ValueError("Detected marker has zero pixel area.")

        scale_cm_per_pixel = known_marker_size_cm / avg_side_px

        warnings = []
        # Check aspect ratio distortion (perspective tilt)
        if avg_pixel_height > 0:
            aspect_ratio = avg_pixel_width / avg_pixel_height
            if aspect_ratio < 0.85 or aspect_ratio > 1.15:
                warnings.append(
                    f"Marker aspect ratio ({aspect_ratio:.2f}) indicates significant perspective tilt. "
                    "Planar scale approximation may have increased error."
                )

        return float(scale_cm_per_pixel), warnings

    def infer_aspect_depth(
        self,
        category: str,
        measured_long_cm: float,
        measured_short_cm: float,
        priors: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[Optional[float], Optional[str], List[str]]:
        """
        Infers the missing orthogonal depth dimension using category prior aspect-ratio proportions.

        Mathematical Rationale:
          A single 2D frontal view measures 2 orthogonal dimensions on the planar reference plane.
          By matching the observed 2D aspect ratio (measured_long / measured_short) to the canonical
          3D proportions (length, width, height) of the category, we derive a conservative estimate
          for the unobserved depth axis without claiming full 3D sensor reconstruction.
        """
        if priors is None:
            priors = DEFAULT_CATEGORY_PRIORS

        cat_key = category.strip().lower()
        if cat_key not in priors:
            return None, None, [f"Cannot infer depth: Category '{category}' has no configured prior envelope."]

        prior = priors[cat_key]
        p_l = float(prior["length_cm"])
        p_w = float(prior["width_cm"])
        p_h = float(prior["height_cm"])

        if p_l <= 0 or p_w <= 0 or p_h <= 0 or measured_short_cm <= 0 or measured_long_cm <= 0:
            return None, None, ["Invalid dimensions for aspect ratio depth inference."]

        # 3 potential planar view pairings from canonical (L, W, H):
        # 1. Front/Back view: Length x Height (missing: Width)
        # 2. Top/Bottom view: Length x Width (missing: Height)
        # 3. Side view: Width x Height (missing: Length)
        view_configs = [
            {
                "name": "front_view",
                "dim1": max(p_l, p_h),
                "dim2": min(p_l, p_h),
                "missing_dim_val": p_w,
                "missing_name": "width",
            },
            {
                "name": "top_view",
                "dim1": max(p_l, p_w),
                "dim2": min(p_l, p_w),
                "missing_dim_val": p_h,
                "missing_name": "height",
            },
            {
                "name": "side_view",
                "dim1": max(p_w, p_h),
                "dim2": min(p_w, p_h),
                "missing_dim_val": p_l,
                "missing_name": "length",
            },
        ]

        obs_ratio = measured_long_cm / measured_short_cm

        # Find best matching view orientation by comparing aspect ratios
        best_config = None
        min_discrepancy = float("inf")
        for cfg in view_configs:
            cfg_ratio = cfg["dim1"] / cfg["dim2"]
            discrepancy = abs(np.log(obs_ratio) - np.log(cfg_ratio))
            if discrepancy < min_discrepancy:
                min_discrepancy = discrepancy
                best_config = cfg

        if best_config is None:
            return None, None, ["Failed to match aspect ratio."]

        # Scale factor from measured dimensions to prior canonical size
        scale_factor = 0.5 * ((measured_long_cm / best_config["dim1"]) + (measured_short_cm / best_config["dim2"]))
        inferred_depth = round(scale_factor * best_config["missing_dim_val"], 1)

        warnings = [
            f"Orthogonal depth ({inferred_depth} cm) inferred from canonical 3D aspect ratio of '{cat_key}' ({best_config['name']}).",
            "Depth is NOT directly measured by the sensor; actual physical depth may vary.",
        ]

        return inferred_depth, "category_aspect_ratio_prior", warnings

    def measure_dimensions(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        known_marker_size_cm: float,
        object_bbox_px: Tuple[int, int, int, int],
        category: Optional[str] = None,
        infer_aspect_depth: bool = False,
    ) -> DimensionEstimateResult:
        """
        Executes reference-marker measurement on the target object bounding box.
        
        Args:
            image_input: File path, numpy array, or PIL Image.
            known_marker_size_cm: Known side length of square marker (cm).
            object_bbox_px: (xmin, ymin, xmax, ymax) of the detected object in pixels.
            category: Optional object class name for depth ratio priors.
            infer_aspect_depth: When True and category is known, infers the missing orthogonal depth axis.
        """
        if known_marker_size_cm <= 0:
            return DimensionEstimateResult(
                status=EstimationStatus.ERROR.value,
                source=None,
                estimation_method=EstimationMethod.NONE.value,
                dimensions_cm={"length": None, "width": None, "height": None},
                confidence=0.0,
                measurement_confidence=0.0,
                scale_cm_per_pixel=None,
                reference={"type": "aruco", "size_cm": known_marker_size_cm, "detected": False},
                warnings=[f"Invalid marker physical size: {known_marker_size_cm} cm. Must be positive."],
            )

        xmin, ymin, xmax, ymax = object_bbox_px
        obj_w_px = max(0, xmax - xmin)
        obj_h_px = max(0, ymax - ymin)

        if obj_w_px <= 0 or obj_h_px <= 0:
            return DimensionEstimateResult(
                status=EstimationStatus.ERROR.value,
                source=None,
                estimation_method=EstimationMethod.NONE.value,
                dimensions_cm={"length": None, "width": None, "height": None},
                confidence=0.0,
                measurement_confidence=0.0,
                scale_cm_per_pixel=None,
                reference={"type": "aruco", "size_cm": known_marker_size_cm, "detected": False},
                warnings=["Invalid object bounding box: width and height must be positive."],
            )

        try:
            detected, corners, marker_id = self.detect_marker(image_input)
        except Exception as e:
            return DimensionEstimateResult(
                status=EstimationStatus.ERROR.value,
                source=None,
                estimation_method=EstimationMethod.NONE.value,
                dimensions_cm={"length": None, "width": None, "height": None},
                confidence=0.0,
                measurement_confidence=0.0,
                scale_cm_per_pixel=None,
                reference={"type": "aruco", "size_cm": known_marker_size_cm, "detected": False},
                warnings=[f"Failed to process image for marker detection: {str(e)}"],
            )

        if not detected or corners is None:
            return DimensionEstimateResult(
                status=EstimationStatus.REFERENCE_REQUIRED.value,
                source=None,
                estimation_method=EstimationMethod.NONE.value,
                dimensions_cm={"length": None, "width": None, "height": None},
                confidence=0.0,
                measurement_confidence=0.0,
                scale_cm_per_pixel=None,
                reference={"type": "aruco", "size_cm": known_marker_size_cm, "detected": False},
                warnings=[
                    "No valid ArUco reference marker detected in image.",
                    "Metric physical dimensions cannot be measured without a calibrated scale reference.",
                ],
            )

        # Calculate scale
        scale_cm_per_px, scale_warnings = self.calculate_scale_cm_per_pixel(corners, known_marker_size_cm)

        # Planar dimensions
        measured_dim_x = obj_w_px * scale_cm_per_px
        measured_dim_y = obj_h_px * scale_cm_per_px

        length_cm = max(measured_dim_x, measured_dim_y)
        height_cm = min(measured_dim_x, measured_dim_y)
        width_cm = None

        warnings = list(scale_warnings)
        warnings.append(
            "Measured dimensions are planar 2D projections calibrated via the reference marker plane."
        )

        depth_method = None
        estimation_method = EstimationMethod.ARUCO_REFERENCE.value
        source = MeasurementSource.ARUCO_REFERENCE.value
        conf = 0.90

        # Deterministic optical planar uncertainty (+/- 3.0% on calibrated reference plane)
        uncertainty_percent = {"length": 3.0, "width": None, "height": 3.0}
        uncertainty_cm = {
            "length": round(length_cm * 0.03, 1),
            "width": None,
            "height": round(height_cm * 0.03, 1),
        }

        # Attempt Aspect-Ratio Depth Inference if requested and category is available
        if infer_aspect_depth and category is not None:
            depth_val, d_method, d_warnings = self.infer_aspect_depth(
                category=category,
                measured_long_cm=length_cm,
                measured_short_cm=height_cm,
            )
            if depth_val is not None:
                width_cm = depth_val
                depth_method = d_method
                estimation_method = EstimationMethod.ARUCO_PLUS_ASPECT_PRIOR.value
                source = MeasurementSource.ARUCO_PLUS_ASPECT_PRIOR.value
                conf = 0.85
                uncertainty_percent["width"] = 20.0
                uncertainty_cm["width"] = round(width_cm * 0.20, 1)
                warnings.extend(d_warnings)
            else:
                warnings.extend(d_warnings)
                warnings.append("Orthogonal depth (width) remains unmeasured from a single monocular view.")
        else:
            warnings.append("Orthogonal depth (width) is unmeasured from a single monocular view.")

        # Default category weight if category known
        weight_val = None
        weight_src = None
        if category is not None and category.strip().lower() in DEFAULT_CATEGORY_PRIORS:
            prior_entry = DEFAULT_CATEGORY_PRIORS[category.strip().lower()]
            weight_val = prior_entry.get("default_weight_kg")
            weight_src = "category_prior"

        return DimensionEstimateResult(
            status=EstimationStatus.MEASURED.value,
            source=source,
            estimation_method=estimation_method,
            dimensions_cm={
                "length": round(length_cm, 1),
                "width": round(width_cm, 1) if width_cm is not None else None,
                "height": round(height_cm, 1),
            },
            confidence=conf,
            measurement_confidence=conf,
            scale_cm_per_pixel=round(scale_cm_per_px, 6),
            weight_kg=round(weight_val, 2) if weight_val is not None else None,
            weight_source=weight_src,
            depth_estimation_method=depth_method,
            uncertainty_percent=uncertainty_percent,
            uncertainty_cm=uncertainty_cm,
            reference={
                "type": "aruco_dict_4x4_50",
                "marker_id": marker_id,
                "size_cm": known_marker_size_cm,
                "detected": True,
                "detected_corners_px": [[round(float(c[0]), 1), round(float(c[1]), 1)] for c in corners],
            },
            warnings=warnings,
        )


# -----------------------------------------------------------------------------
# Unified Hybrid Dimension Estimator
# -----------------------------------------------------------------------------

class DimensionEstimator:
    """
    Unified Hybrid Dimension Estimation Engine.
    Orchestrates Reference-Marker Measurement (Mode 1), Aspect-Ratio Depth Inference (Mode 1b),
    and Category-Prior Fallback (Mode 2).
    """

    def __init__(
        self,
        priors: Optional[Dict[str, Dict[str, Any]]] = None,
        aruco_dict_type: int = cv2.aruco.DICT_4X4_50,
    ):
        self.reference_estimator = ReferenceMarkerEstimator(aruco_dict_type=aruco_dict_type)
        self.prior_estimator = CategoryPriorEstimator(priors=priors)

    def estimate(
        self,
        image_input: Optional[Union[str, np.ndarray, Image.Image]] = None,
        category: Optional[str] = None,
        known_marker_size_cm: Optional[float] = None,
        object_bbox_px: Optional[Tuple[int, int, int, int]] = None,
        fallback_to_prior: bool = True,
        infer_aspect_depth: bool = False,
    ) -> DimensionEstimateResult:
        """
        Main entry point for physical dimension estimation.
        
        Workflow:
          1. If image, marker size, and bbox are provided, attempts Reference-Marker Measurement.
          2. If marker is missing/unusable and fallback_to_prior=True and category is given, returns Category Prior.
          3. Otherwise returns REFERENCE_REQUIRED or ERROR.
        """
        # 1. Attempt Mode 1 (Reference Measurement) if parameters provided
        if image_input is not None and known_marker_size_cm is not None and object_bbox_px is not None:
            res = self.reference_estimator.measure_dimensions(
                image_input=image_input,
                known_marker_size_cm=known_marker_size_cm,
                object_bbox_px=object_bbox_px,
                category=category,
                infer_aspect_depth=infer_aspect_depth,
            )
            if res.status == EstimationStatus.MEASURED.value:
                return res
            elif res.status == EstimationStatus.REFERENCE_REQUIRED.value and fallback_to_prior and category:
                # Fallback to Mode 2 with combined warning
                prior_res = self.prior_estimator.estimate(category)
                prior_res.warnings.insert(
                    0, "Reference marker was NOT detected. Falling back to category-prior estimate."
                )
                return prior_res
            else:
                return res

        # 2. Mode 2 direct execution (if no marker measurement requested)
        if category is not None:
            return self.prior_estimator.estimate(category)

        # 3. Insufficient parameters
        return DimensionEstimateResult(
            status=EstimationStatus.ERROR.value,
            source=None,
            estimation_method=EstimationMethod.NONE.value,
            dimensions_cm={"length": None, "width": None, "height": None},
            confidence=0.0,
            measurement_confidence=0.0,
            scale_cm_per_pixel=None,
            reference={"type": None, "size_cm": None, "detected": False},
            warnings=["Insufficient input parameters: provide either (image, marker_size, bbox) or a valid category."],
        )
