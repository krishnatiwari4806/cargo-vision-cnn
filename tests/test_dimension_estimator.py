"""
test_dimension_estimator.py - Unit Tests for Physical Dimension Estimation Engine
==================================================================================
Project: Cargo Vision Logistics System (Phase 2)
Purpose: Tests Reference-Marker Measurement (Mode 1), Category Prior Fallback (Mode 2),
         error handling, schema compliance, and geometric boundary conditions.
"""

import os
import sys
import unittest
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.dimension_estimator import (
    DimensionEstimator,
    ReferenceMarkerEstimator,
    CategoryPriorEstimator,
    DimensionEstimateResult,
    EstimationStatus,
    MeasurementSource,
    DEFAULT_CATEGORY_PRIORS,
)


class TestDimensionEstimator(unittest.TestCase):
    """
    Comprehensive test suite for dimension estimation engine.
    """

    @classmethod
    def setUpClass(cls):
        # Create deterministic synthetic test fixture image containing a known ArUco marker
        cls.marker_size_px = 100
        cls.marker_size_cm = 10.0  # Expected scale: 10.0 / 100.0 = 0.10 cm/px
        cls.expected_scale = cls.marker_size_cm / cls.marker_size_px

        # Generate ArUco marker ID 0 (4x4_50)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, cls.marker_size_px)

        # Create 600x600 white canvas
        cls.canvas_with_marker = np.ones((600, 600, 3), dtype=np.uint8) * 255
        # Place marker at (50, 50) -> (150, 150)
        cls.canvas_with_marker[50:150, 50:150] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

        # Blank canvas without marker
        cls.canvas_without_marker = np.ones((600, 600, 3), dtype=np.uint8) * 255

    def setUp(self):
        self.ref_estimator = ReferenceMarkerEstimator()
        self.prior_estimator = CategoryPriorEstimator()
        self.unified_estimator = DimensionEstimator()

    # -------------------------------------------------------------------------
    # 1. Valid Marker Scale Calculation
    # -------------------------------------------------------------------------
    def test_valid_marker_scale_and_measurement(self):
        """Verify exact scale calculation and dimension measurement with valid marker fixture."""
        # Simulated object bbox: 400px wide, 200px high
        obj_bbox = (200, 200, 600, 400)  # w=400px, h=200px

        result = self.ref_estimator.measure_dimensions(
            image_input=self.canvas_with_marker,
            known_marker_size_cm=self.marker_size_cm,
            object_bbox_px=obj_bbox,
        )

        self.assertEqual(result.status, EstimationStatus.MEASURED.value)
        self.assertEqual(result.source, MeasurementSource.ARUCO_REFERENCE.value)
        self.assertIsNotNone(result.scale_cm_per_pixel)
        self.assertAlmostEqual(result.scale_cm_per_pixel, self.expected_scale, delta=0.005)

        # Expected dimensions: 400px * 0.1cm/px = 40.0cm (Length), 200px * 0.1cm/px = 20.0cm (Height)
        self.assertAlmostEqual(result.dimensions_cm["length"], 40.0, delta=1.0)
        self.assertAlmostEqual(result.dimensions_cm["height"], 20.0, delta=1.0)
        self.assertIsNone(result.dimensions_cm["width"])  # Orthogonal depth unmeasured in 2D
        self.assertTrue(result.reference["detected"])
        self.assertEqual(result.reference["marker_id"], 0)

    # -------------------------------------------------------------------------
    # 2. Invalid Marker Size Handling
    # -------------------------------------------------------------------------
    def test_invalid_marker_size(self):
        """Verify negative and zero marker sizes return ERROR and do not compute scale."""
        res_zero = self.ref_estimator.measure_dimensions(
            image_input=self.canvas_with_marker,
            known_marker_size_cm=0.0,
            object_bbox_px=(10, 10, 50, 50),
        )
        self.assertEqual(res_zero.status, EstimationStatus.ERROR.value)
        self.assertIsNone(res_zero.scale_cm_per_pixel)
        self.assertIsNone(res_zero.dimensions_cm["length"])

        res_neg = self.ref_estimator.measure_dimensions(
            image_input=self.canvas_with_marker,
            known_marker_size_cm=-10.0,
            object_bbox_px=(10, 10, 50, 50),
        )
        self.assertEqual(res_neg.status, EstimationStatus.ERROR.value)

    # -------------------------------------------------------------------------
    # 3. Missing Marker Handling
    # -------------------------------------------------------------------------
    def test_missing_marker_returns_reference_required(self):
        """Verify missing marker returns REFERENCE_REQUIRED without fabricating dimensions."""
        result = self.ref_estimator.measure_dimensions(
            image_input=self.canvas_without_marker,
            known_marker_size_cm=10.0,
            object_bbox_px=(100, 100, 300, 300),
        )
        self.assertEqual(result.status, EstimationStatus.REFERENCE_REQUIRED.value)
        self.assertIsNone(result.source)
        self.assertIsNone(result.scale_cm_per_pixel)
        self.assertIsNone(result.dimensions_cm["length"])
        self.assertIsNone(result.dimensions_cm["height"])
        self.assertFalse(result.reference["detected"])

    # -------------------------------------------------------------------------
    # 4. Invalid Image Input
    # -------------------------------------------------------------------------
    def test_invalid_image_path(self):
        """Verify non-existent image path returns ERROR."""
        result = self.ref_estimator.measure_dimensions(
            image_input="non_existent_file_path_123.jpg",
            known_marker_size_cm=10.0,
            object_bbox_px=(10, 10, 50, 50),
        )
        self.assertEqual(result.status, EstimationStatus.ERROR.value)

    # -------------------------------------------------------------------------
    # 5. Category Prior Lookup
    # -------------------------------------------------------------------------
    def test_category_prior_lookup_all_categories(self):
        """Verify all standard categories return PRIOR_ESTIMATE with configured envelopes."""
        categories = ["box", "chair", "couch", "table", "suitcase", "car"]
        for cat in categories:
            res = self.prior_estimator.estimate(cat)
            self.assertEqual(res.status, EstimationStatus.PRIOR_ESTIMATE.value)
            self.assertEqual(res.source, MeasurementSource.CATEGORY_PRIOR.value)
            self.assertGreater(res.dimensions_cm["length"], 0.0)
            self.assertGreater(res.dimensions_cm["width"], 0.0)
            self.assertGreater(res.dimensions_cm["height"], 0.0)
            self.assertIn("standard category priors", " ".join(res.warnings).lower())

    # -------------------------------------------------------------------------
    # 6. Unknown Category
    # -------------------------------------------------------------------------
    def test_unknown_category_returns_error(self):
        """Verify querying an unknown category returns ERROR."""
        res = self.prior_estimator.estimate("unknown_spaceship_item")
        self.assertEqual(res.status, EstimationStatus.ERROR.value)
        self.assertIsNone(res.dimensions_cm["length"])

    # -------------------------------------------------------------------------
    # 7. Result Schema Integrity
    # -------------------------------------------------------------------------
    def test_result_schema_and_serialization(self):
        """Verify DimensionEstimateResult produces required fields and serializes to dict."""
        res = self.prior_estimator.estimate("couch")
        res_dict = res.to_dict()

        required_keys = ["status", "source", "dimensions_cm", "confidence", "scale_cm_per_pixel", "reference", "warnings"]
        for key in required_keys:
            self.assertIn(key, res_dict, f"Missing key in schema: {key}")

        self.assertIn("length", res_dict["dimensions_cm"])
        self.assertIn("width", res_dict["dimensions_cm"])
        self.assertIn("height", res_dict["dimensions_cm"])

    # -------------------------------------------------------------------------
    # 8. No Fabricated Measurements When Scale Unavailable
    # -------------------------------------------------------------------------
    def test_no_fabricated_measurements_on_blank_image(self):
        """Verify unified estimator without fallback returns REFERENCE_REQUIRED and zero metric numbers."""
        res = self.unified_estimator.estimate(
            image_input=self.canvas_without_marker,
            known_marker_size_cm=10.0,
            object_bbox_px=(50, 50, 200, 200),
            fallback_to_prior=False,
        )
        self.assertEqual(res.status, EstimationStatus.REFERENCE_REQUIRED.value)
        self.assertIsNone(res.dimensions_cm["length"])
        self.assertIsNone(res.dimensions_cm["height"])

    # -------------------------------------------------------------------------
    # 9. Pixel to Scale Math Verification
    # -------------------------------------------------------------------------
    def test_pixel_to_scale_direct_calculation(self):
        """Verify calculate_scale_cm_per_pixel mathematical correctness."""
        corners = np.array([
            [100.0, 100.0],
            [200.0, 100.0],
            [200.0, 200.0],
            [100.0, 200.0],
        ])  # 100px square
        scale, warnings = self.ref_estimator.calculate_scale_cm_per_pixel(corners, known_marker_size_cm=20.0)
        # Expected scale: 20.0 cm / 100.0 px = 0.20 cm/px
        self.assertAlmostEqual(scale, 0.20, places=5)
        self.assertEqual(len(warnings), 0)

    # -------------------------------------------------------------------------
    # 10. Geometry & Bounding Box Validation
    # -------------------------------------------------------------------------
    def test_invalid_object_bounding_box(self):
        """Verify inverted or zero-area bounding boxes return ERROR."""
        # Zero area bbox
        res_zero = self.ref_estimator.measure_dimensions(
            image_input=self.canvas_with_marker,
            known_marker_size_cm=10.0,
            object_bbox_px=(100, 100, 100, 100),
        )
        self.assertEqual(res_zero.status, EstimationStatus.ERROR.value)

        # Inverted coordinates
        res_inv = self.ref_estimator.measure_dimensions(
            image_input=self.canvas_with_marker,
            known_marker_size_cm=10.0,
            object_bbox_px=(200, 200, 100, 100),
        )
        self.assertEqual(res_inv.status, EstimationStatus.ERROR.value)


if __name__ == "__main__":
    unittest.main()
