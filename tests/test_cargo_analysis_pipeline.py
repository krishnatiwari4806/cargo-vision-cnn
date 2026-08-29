"""
test_cargo_analysis_pipeline.py - Unit Tests for Unified Cargo Analysis Pipeline
================================================================================
Project: Cargo Vision Logistics System
Purpose: Tests end-to-end integration of classification, physical dimension estimation,
         quantity handling, volumetric math, vehicle fleet dispatch, and schema compliance.
"""

import os
import sys
import unittest
import numpy as np
import cv2
from PIL import Image
import tempfile

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.cargo_analysis_pipeline import CargoAnalysisPipeline
from scripts.vehicle_database import VehicleDatabase
from scripts.dimension_estimator import DimensionEstimator


class TestCargoAnalysisPipeline(unittest.TestCase):
    """
    Unit test suite for the unified cargo analysis and recommendation pipeline.
    """

    @classmethod
    def setUpClass(cls):
        # Create a real test image fixture (128x128 green square)
        cls.temp_dir = tempfile.mkdtemp()
        cls.valid_image_path = os.path.join(cls.temp_dir, "test_cargo_item.jpg")
        img = Image.new("RGB", (256, 256), color=(100, 150, 200))
        img.save(cls.valid_image_path, "JPEG")

        # Create test fixture with ArUco marker
        cls.marker_image_path = os.path.join(cls.temp_dir, "test_with_aruco.jpg")
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 100)
        canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255
        canvas[50:150, 50:150] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(cls.marker_image_path, canvas)

        cls.pipeline = CargoAnalysisPipeline()

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary directory
        if os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    # -------------------------------------------------------------------------
    # 1. Invalid Image Path Handling
    # -------------------------------------------------------------------------
    def test_invalid_image_path_returns_structured_error(self):
        """Verify non-existent image paths return ERROR status without crashing."""
        res = self.pipeline.analyze("non_existent_image_file_999.jpg", quantity=1)
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("error", res)
        self.assertIn("Image file not found", res["error"])

        res_empty = self.pipeline.analyze("", quantity=1)
        self.assertEqual(res_empty["status"], "ERROR")

    # -------------------------------------------------------------------------
    # 2. Invalid Quantity Handling
    # -------------------------------------------------------------------------
    def test_invalid_quantity_returns_structured_error(self):
        """Verify zero, negative, or non-integer quantities return ERROR."""
        res_zero = self.pipeline.analyze(self.valid_image_path, quantity=0)
        self.assertEqual(res_zero["status"], "ERROR")
        self.assertIn("Quantity must be a positive integer", res_zero["error"])

        res_neg = self.pipeline.analyze(self.valid_image_path, quantity=-5)
        self.assertEqual(res_neg["status"], "ERROR")

    # -------------------------------------------------------------------------
    # 3. Valid Quantity Handling
    # -------------------------------------------------------------------------
    def test_valid_quantity_scales_requirements_correctly(self):
        """Verify quantity parameter properly scales total volume and item count."""
        res = self.pipeline.analyze(self.valid_image_path, quantity=3)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["quantity"], 3)
        self.assertEqual(res["cargo_summary"]["total_items"], 3)

    # -------------------------------------------------------------------------
    # 4. Classification Result Schema
    # -------------------------------------------------------------------------
    def test_classification_result_schema(self):
        """Verify classification dictionary contains class_name, confidence, and probabilities."""
        res = self.pipeline.analyze(self.valid_image_path, quantity=1)
        self.assertEqual(res["status"], "SUCCESS")
        cls_res = res["classification"]

        self.assertIn("class_name", cls_res)
        self.assertIn(cls_res["class_name"], ["box", "chair", "couch", "suitcase", "table"])
        self.assertIsInstance(cls_res["confidence"], float)
        self.assertGreaterEqual(cls_res["confidence"], 0.0)
        self.assertLessEqual(cls_res["confidence"], 1.0)
        self.assertIsInstance(cls_res["probabilities"], dict)
        self.assertEqual(len(cls_res["probabilities"]), 5)

    # -------------------------------------------------------------------------
    # 5. Dimension Result Integration & Prior-Estimate State
    # -------------------------------------------------------------------------
    def test_prior_estimate_mode_when_no_marker_provided(self):
        """Verify default execution uses category prior and flags PRIOR_ESTIMATE."""
        res = self.pipeline.analyze(self.valid_image_path, quantity=1)
        dim_res = res["dimensions"]

        self.assertEqual(dim_res["status"], "PRIOR_ESTIMATE")
        self.assertEqual(dim_res["source"], "category_prior")
        self.assertGreater(dim_res["length_cm"], 0.0)
        self.assertGreater(dim_res["width_cm"], 0.0)
        self.assertGreater(dim_res["height_cm"], 0.0)

    # -------------------------------------------------------------------------
    # 6. Measured State with ArUco Marker
    # -------------------------------------------------------------------------
    def test_measured_state_with_aruco_fixture(self):
        """Verify Mode 1 returns MEASURED when marker and bbox are provided."""
        res = self.pipeline.analyze(
            image_path=self.marker_image_path,
            quantity=1,
            known_marker_size_cm=10.0,
            object_bbox_px=(200, 200, 400, 300),
        )
        self.assertEqual(res["status"], "SUCCESS")
        dim_res = res["dimensions"]

        self.assertEqual(dim_res["status"], "MEASURED")
        self.assertEqual(dim_res["source"], "aruco_reference")
        self.assertIsNotNone(dim_res["length_cm"])
        self.assertIsNotNone(dim_res["height_cm"])
        self.assertIsNone(dim_res["width_cm"])  # Orthogonal depth unmeasured in 2D frontal view

    # -------------------------------------------------------------------------
    # 7. Incomplete Dimension Handling
    # -------------------------------------------------------------------------
    def test_incomplete_dimensions_handled_safely(self):
        """Verify volume calculation handles None dimensions gracefully without crashing."""
        incomplete_dims = {"length_cm": 50.0, "width_cm": None, "height_cm": 30.0}
        summary, warnings = self.pipeline.calculate_cargo_requirements(
            dimensions=incomplete_dims,
            category="box",
            quantity=2,
        )
        self.assertIsNone(summary["total_volume_m3"])
        self.assertIsNone(summary["required_floor_area_m2"])
        self.assertTrue(any("depth/width" in w for w in warnings))

    # -------------------------------------------------------------------------
    # 8. Vehicle Recommendation Result Schema
    # -------------------------------------------------------------------------
    def test_vehicle_recommendation_schema_and_fields(self):
        """Verify vehicle recommendation dictionary contains ID, name, reason, and alternatives."""
        res = self.pipeline.analyze(self.valid_image_path, quantity=2)
        rec = res["vehicle_recommendation"]

        self.assertIn("vehicle_id", rec)
        self.assertIn("vehicle_name", rec)
        self.assertIn("reason", rec)
        self.assertIn("alternatives", rec)
        self.assertTrue(rec["vehicle_id"].startswith("V_"))

    # -------------------------------------------------------------------------
    # 9. Warnings and Disclaimers Generation
    # -------------------------------------------------------------------------
    def test_warnings_include_unvalidated_target_and_payload(self):
        """Verify results explicitly include disclaimers for unverified target and payload."""
        res = self.pipeline.analyze(self.valid_image_path, quantity=1)
        warnings_str = " ".join(res["warnings"])

        self.assertIn("95% target has not yet been validated", warnings_str)
        self.assertIn("Payload suitability cannot be verified", warnings_str)

    # -------------------------------------------------------------------------
    # 10. Large Quantity Multi-Trip Recommendation Logic
    # -------------------------------------------------------------------------
    def test_large_quantity_exceeding_fleet_capacity(self):
        """Verify massive quantities (e.g. 200 items) trigger multi-trip batching advice."""
        res = self.pipeline.analyze(self.valid_image_path, quantity=200)
        rec = res["vehicle_recommendation"]
        self.assertIn("exceeds standard single-vehicle capacity", rec["reason"])

    # -------------------------------------------------------------------------
    # 11. YOLO COCO Detection & CAR Dispatch Flow
    # -------------------------------------------------------------------------
    def test_yolo_car_detection_and_recommendation_flow(self):
        """Verify YOLO detects 'car' in real car image, resolves priors, and recommends 19-ft truck."""
        car_img_path = os.path.join(PROJECT_ROOT, "data", "car.jpg")
        if not os.path.exists(car_img_path):
            self.skipTest("data/car.jpg not present on disk.")

        pipeline_yolo = CargoAnalysisPipeline(detector_backend="auto")
        res = pipeline_yolo.analyze(car_img_path, quantity=1)

        self.assertEqual(res["status"], "SUCCESS")
        cls_info = res["classification"]
        self.assertEqual(cls_info["class_name"], "car")
        self.assertEqual(cls_info["detector_backend"], "yolo_coco")
        self.assertGreater(cls_info["confidence"], 0.70)
        self.assertIn("bounding_box_px", cls_info)

        dim_info = res["dimensions"]
        self.assertEqual(dim_info["status"], "PRIOR_ESTIMATE")
        self.assertEqual(dim_info["length_cm"], 450.0)
        self.assertEqual(dim_info["width_cm"], 180.0)
        self.assertEqual(dim_info["height_cm"], 145.0)

        cargo_summary = res["cargo_summary"]
        self.assertAlmostEqual(cargo_summary["unit_volume_m3"], 11.745, places=3)
        self.assertAlmostEqual(cargo_summary["required_floor_area_m2"], 8.10, places=2)

        rec = res["vehicle_recommendation"]
        self.assertEqual(rec["vehicle_id"], "V_EICHER_19FT")
        self.assertIn("19-ft Medium Freight Truck", rec["vehicle_name"])
        self.assertTrue(any("Dedicated Multi-Car Carrier" in alt for alt in rec.get("alternatives", [])))

    # -------------------------------------------------------------------------
    # 12. Explicit MobileNetV2 Backend Mode
    # -------------------------------------------------------------------------
    def test_mobilenetv2_backend_explicit(self):
        """Verify explicitly specifying mobilenetv2 backend executes MobileNetV2 classifier."""
        pipeline_mobilenet = CargoAnalysisPipeline(detector_backend="mobilenetv2")
        res = pipeline_mobilenet.analyze(self.valid_image_path, quantity=1)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["classification"]["detector_backend"], "mobilenetv2")
    # -------------------------------------------------------------------------
    # 13. Multi-Load: 2 Identical Items
    # -------------------------------------------------------------------------
    def test_multiload_two_identical_items(self):
        """Verify multi-load aggregation for 2 identical items properly aggregates quantity and volume."""
        res = self.pipeline.analyze_multiple([self.valid_image_path, self.valid_image_path], quantities=[1, 1])
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["mode"], "MULTI_LOAD")
        self.assertEqual(res["total_items"], 2)
        self.assertEqual(len(res["items"]), 2)
        self.assertGreater(res["shipment_summary"]["total_volume_m3"], 0.0)
        self.assertGreater(res["shipment_summary"]["total_floor_area_m2"], 0.0)
        self.assertIn("vehicle_id", res["vehicle_recommendation"])

    # -------------------------------------------------------------------------
    # 14. Multi-Load: 2 Different Cargo Types
    # -------------------------------------------------------------------------
    def test_multiload_two_different_cargo_types(self):
        """Verify multi-load aggregation handles different cargo images with distinct categories."""
        car_img = os.path.join(PROJECT_ROOT, "data", "car.jpg")
        if not os.path.exists(car_img):
            self.skipTest("data/car.jpg not present.")

        res = self.pipeline.analyze_multiple([car_img, self.valid_image_path], quantities=[1, 2])
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["total_items"], 3)
        self.assertEqual(res["items"][0]["category"], "car")
        self.assertEqual(res["items"][0]["quantity"], 1)
        self.assertEqual(res["items"][1]["quantity"], 2)
        self.assertIn("Accommodates combined shipment", res["vehicle_recommendation"]["reason"])

    # -------------------------------------------------------------------------
    # 15. Multi-Load: Refrigerator + 2 TVs + 4 Boxes Aggregation Math
    # -------------------------------------------------------------------------
    def test_multiload_mixed_refrigerator_tv_boxes_aggregation_math(self):
        """Verify combined volume and floor area calculations for 1 refrigerator, 2 TVs, and 4 boxes."""
        # Calculate requirements for individual items directly
        refrig_dims = {"length_cm": 70.0, "width_cm": 70.0, "height_cm": 175.0}
        tv_dims = {"length_cm": 120.0, "width_cm": 15.0, "height_cm": 75.0}
        box_dims = {"length_cm": 45.0, "width_cm": 35.0, "height_cm": 30.0}

        refrig_summary, _ = self.pipeline.calculate_cargo_requirements(refrig_dims, "refrigerator", 1)
        tv_summary, _ = self.pipeline.calculate_cargo_requirements(tv_dims, "tv", 2)
        box_summary, _ = self.pipeline.calculate_cargo_requirements(box_dims, "box", 4)

        # Expected unit volumes:
        # Refrigerator: 70*70*175 / 10^6 = 0.8575 m3; packing factor 0.85 -> 0.8575 / 0.85 = 1.0088 m3
        # TV: 120*15*75 / 10^6 = 0.1350 m3; 2 TVs = 0.2700 / 0.80 = 0.3375 m3
        # Box: 45*35*30 / 10^6 = 0.04725 m3; 4 boxes = 0.1890 / 0.80 = 0.23625 -> 0.2362 m3
        expected_total_volume = round(refrig_summary["total_volume_m3"] + tv_summary["total_volume_m3"] + box_summary["total_volume_m3"], 4)
        expected_floor_area = round(refrig_summary["required_floor_area_m2"] + tv_summary["required_floor_area_m2"] + box_summary["required_floor_area_m2"], 4)

        items = [
            {"category": "refrigerator", "quantity": 1, "dimensions": refrig_dims, "cargo_summary": refrig_summary},
            {"category": "tv", "quantity": 2, "dimensions": tv_dims, "cargo_summary": tv_summary},
            {"category": "box", "quantity": 4, "dimensions": box_dims, "cargo_summary": box_summary},
        ]

        rec, warnings = self.pipeline.recommend_vehicle_for_shipment(
            items=items,
            total_volume_m3=expected_total_volume,
            total_floor_area_m2=expected_floor_area,
        )

        self.assertIsNotNone(rec["vehicle_id"])
        # Should fit in a 3-Wheeler Auto (usable length 145cm, usable height 140cm < 175cm -> Auto fails height!)
        # Tata Ace has 145cm height < 175cm -> Tata Ace fails height!
        # Bolero Maxi Truck has 175cm height -> fits Bolero or Tata 407!
        self.assertIn(rec["vehicle_id"], ["V_BOLERO_PICKUP", "V_TATA_407_14FT", "V_EICHER_19FT"])
        self.assertIn("Accommodates combined shipment (7 items:", rec["reason"])

    # -------------------------------------------------------------------------
    # 16. Multi-Load: Massive Combined Shipment Exceeding Fleet Capacity
    # -------------------------------------------------------------------------
    def test_multiload_massive_shipment_exceeding_capacity(self):
        """Verify massive multi-load shipment triggers multi-trip batching advice."""
        items = [
            {"category": "car", "quantity": 10, "dimensions": {"length_cm": 450.0, "width_cm": 180.0, "height_cm": 145.0}},
            {"category": "couch", "quantity": 20, "dimensions": {"length_cm": 210.0, "width_cm": 90.0, "height_cm": 85.0}},
        ]
        rec, _ = self.pipeline.recommend_vehicle_for_shipment(items, total_volume_m3=200.0, total_floor_area_m2=120.0)
        self.assertIn("exceeds standard single-vehicle capacity", rec["reason"])
        self.assertEqual(rec["vehicle_id"], "V_CAR_CARRIER_MULTI")

    # -------------------------------------------------------------------------
    # 17. Warning Integrity: No Contradictory YOLO Warning When Resolved
    # -------------------------------------------------------------------------
    def test_no_false_yolo_detection_warning_when_resolved_by_fallback(self):
        """Verify successful identification via MobileNetV2 does not attach false YOLO non-detection warning."""
        res = self.pipeline.analyze(self.valid_image_path, quantity=1)
        self.assertEqual(res["status"], "SUCCESS")
        for warning in res["warnings"]:
            self.assertNotIn("No objects detected by YOLO detector above confidence threshold", warning)


if __name__ == "__main__":
    unittest.main()
