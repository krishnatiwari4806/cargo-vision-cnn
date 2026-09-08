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

    # -------------------------------------------------------------------------
    # 18. Payload Weight: Small Volume but Heavy Weight Upgrades Vehicle
    # -------------------------------------------------------------------------
    def test_payload_weight_constraint_upgrades_overloaded_vehicle(self):
        """Verify a shipment with small volume but payload exceeding vehicle limit upgrades to a higher payload vehicle."""
        # Dimensions fit 3-Wheeler Auto (145x130x120 cm, 2.26 m3, 1.88 m2 bed), but weight = 700 kg > 500 kg limit of Auto
        items = [
            {"category": "box", "quantity": 10, "dimensions": {"length_cm": 50.0, "width_cm": 40.0, "height_cm": 30.0}},
        ]
        # Total volume: 0.5 m3, floor area: 0.5 m2, weight: 700.0 kg
        rec, _ = self.pipeline.recommend_vehicle_for_shipment(
            items=items,
            total_volume_m3=0.5,
            total_floor_area_m2=0.5,
            total_weight_kg=700.0,
        )
        # V_3W_AUTO (max 500 kg) must be rejected due to payload weight.
        # V_TATA_ACE (max 850 kg) must be selected.
        self.assertEqual(rec["vehicle_id"], "V_TATA_ACE")
        self.assertIn("700.0 kg estimated payload", rec["reason"])

    # -------------------------------------------------------------------------
    # 19. Payload Weight: Quantity Scales Total Weight Correctly
    # -------------------------------------------------------------------------
    def test_quantity_scales_payload_weight(self):
        """Verify quantity multiplier correctly scales total payload weight."""
        dims = {"length_cm": 45.0, "width_cm": 35.0, "height_cm": 30.0, "weight_kg": 12.0}
        cargo_sum, _ = self.pipeline.calculate_cargo_requirements(dims, "box", quantity=5)
        self.assertEqual(cargo_sum["unit_weight_kg"], 12.0)
        self.assertEqual(cargo_sum["total_weight_kg"], 60.0)

    # -------------------------------------------------------------------------
    # 20. Payload Weight: Mixed-Load Weight Aggregation
    # -------------------------------------------------------------------------
    def test_mixed_load_payload_weight_aggregation(self):
        """Verify mixed cargo shipment sums individual item weights correctly."""
        # 1 refrigerator (75 kg) + 2 TVs (2*15 = 30 kg) + 4 boxes (4*12 = 48 kg) = 153 kg
        refrig_dims = {"length_cm": 70.0, "width_cm": 70.0, "height_cm": 175.0, "weight_kg": 75.0}
        tv_dims = {"length_cm": 120.0, "width_cm": 15.0, "height_cm": 75.0, "weight_kg": 15.0}
        box_dims = {"length_cm": 45.0, "width_cm": 35.0, "height_cm": 30.0, "weight_kg": 12.0}

        refrig_sum, _ = self.pipeline.calculate_cargo_requirements(refrig_dims, "refrigerator", 1)
        tv_sum, _ = self.pipeline.calculate_cargo_requirements(tv_dims, "tv", 2)
        box_sum, _ = self.pipeline.calculate_cargo_requirements(box_dims, "box", 4)

        expected_total_weight = round(refrig_sum["total_weight_kg"] + tv_sum["total_weight_kg"] + box_sum["total_weight_kg"], 2)
        self.assertEqual(expected_total_weight, 153.0)

    # -------------------------------------------------------------------------
    # 21. Payload Weight: Heavy Shipment Exceeding All Vehicle Payloads
    # -------------------------------------------------------------------------
    def test_heavy_shipment_exceeding_fleet_payload_capacity(self):
        """Verify massive payload weight exceeding fleet maximum returns batching advice indicating payload."""
        items = [
            {"category": "car", "quantity": 20, "dimensions": {"length_cm": 450.0, "width_cm": 180.0, "height_cm": 145.0}},
        ]
        # Total weight 20 * 1400 = 28,000 kg > 20,000 kg max carrier payload
        rec, _ = self.pipeline.recommend_vehicle_for_shipment(
            items=items,
            total_volume_m3=50.0,
            total_floor_area_m2=30.0,
            total_weight_kg=28000.0,
        )
        self.assertEqual(rec["vehicle_id"], "V_CAR_CARRIER_MULTI")
        self.assertIn("payload weight capacity", rec["reason"])

    # -------------------------------------------------------------------------
    # 22. Payload Weight: Honest Weight Disclaimer
    # -------------------------------------------------------------------------
    def test_payload_weight_disclaimer_preserves_honesty(self):
        """Verify disclaimers state that payload weight is estimated and not measured on a scale."""
        res = self.pipeline.analyze(self.valid_image_path, quantity=1)
        warnings_str = " ".join(res["warnings"])
        self.assertIn("Payload suitability cannot be verified", warnings_str)
        self.assertIn("estimated from standard category priors", warnings_str)

    # -------------------------------------------------------------------------
    # 23. Single-Image Multi-Object Extraction on Real Image
    # -------------------------------------------------------------------------
    def test_single_image_multi_object_extraction_real_image(self):
        """Verify extract_all_objects=True extracts all valid cargo objects from a single photo."""
        multi_img_path = os.path.join(PROJECT_ROOT, "deployment_dataset_expanded", "test", "images", "coco_000000057238.jpg")
        if not os.path.exists(multi_img_path):
            self.skipTest("coco_000000057238.jpg not present on disk.")

        pipeline_yolo = CargoAnalysisPipeline(detector_backend="auto")
        res = pipeline_yolo.analyze(multi_img_path, extract_all_objects=True)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["mode"], "SINGLE_IMAGE_MULTI_OBJECT")
        self.assertIn("detected_objects", res)
        self.assertGreaterEqual(len(res["detected_objects"]), 2)

        # Check detected categories include refrigerator, chair, table
        detected_cats = {item["category"] for item in res["detected_objects"]}
        self.assertIn("refrigerator", detected_cats)
        self.assertIn("chair", detected_cats)

        # Total items count should reflect multiple instances (e.g. 3 chairs + 1 refrig + 1 table = 5)
        self.assertGreaterEqual(res["total_items"], 4)
        self.assertGreater(res["shipment_summary"]["total_volume_m3"], 2.0)
        self.assertGreater(res["shipment_summary"]["total_weight_kg"], 100.0)
        self.assertIsNotNone(res["vehicle_recommendation"]["vehicle_id"])

    # -------------------------------------------------------------------------
    # 24. Single-Image Multi-Object: Duplicate Categories Aggregated Correctly
    # -------------------------------------------------------------------------
    def test_single_image_multi_object_duplicate_categories_aggregated(self):
        """Verify multiple instances of the same category in one image are aggregated by quantity."""
        multi_img_path = os.path.join(PROJECT_ROOT, "deployment_dataset_expanded", "test", "images", "coco_000000057238.jpg")
        if not os.path.exists(multi_img_path):
            self.skipTest("coco_000000057238.jpg not present on disk.")

        pipeline_yolo = CargoAnalysisPipeline(detector_backend="auto")
        res = pipeline_yolo.analyze(multi_img_path, extract_all_objects=True)

        chair_entries = [item for item in res["detected_objects"] if item["category"] == "chair"]
        # There should be exactly 1 aggregated chair entry with quantity >= 2
        self.assertEqual(len(chair_entries), 1)
        self.assertGreaterEqual(chair_entries[0]["quantity"], 2)
        self.assertEqual(chair_entries[0]["detected_instances"], chair_entries[0]["quantity"])

    # -------------------------------------------------------------------------
    # 25. Single-Image Multi-Object: Unsupported Non-Cargo Detections Ignored
    # -------------------------------------------------------------------------
    def test_single_image_multi_object_unsupported_detections_ignored(self):
        """Verify non-cargo COCO objects like remotes, cups, dogs, persons are excluded from cargo list."""
        # Image with couch and remote: deployment_dataset_expanded/test/images/coco_000000290771.jpg
        img_path = os.path.join(PROJECT_ROOT, "deployment_dataset_expanded", "test", "images", "coco_000000290771.jpg")
        if not os.path.exists(img_path):
            self.skipTest("coco_000000290771.jpg not present on disk.")

        pipeline_yolo = CargoAnalysisPipeline(detector_backend="auto")
        res = pipeline_yolo.analyze(img_path, extract_all_objects=True)

        self.assertEqual(res["status"], "SUCCESS")
        detected_cats = {item["category"] for item in res["detected_objects"]}
        # Remote or person must NOT appear in cargo objects
        self.assertNotIn("remote", detected_cats)
        self.assertNotIn("person", detected_cats)
        self.assertIn("couch", detected_cats)

    # -------------------------------------------------------------------------
    # 26. Single-Image Multi-Object: Default extract_all_objects=False Preserved
    # -------------------------------------------------------------------------
    def test_single_image_multi_object_default_false_preserves_single_load(self):
        """Verify default extract_all_objects=False returns SINGLE_LOAD mode with primary object."""
        car_img_path = os.path.join(PROJECT_ROOT, "data", "car.jpg")
        if not os.path.exists(car_img_path):
            self.skipTest("data/car.jpg not present on disk.")

        pipeline_yolo = CargoAnalysisPipeline(detector_backend="auto")
        res = pipeline_yolo.analyze(car_img_path, quantity=1, extract_all_objects=False)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["mode"], "SINGLE_LOAD")
        self.assertEqual(res["classification"]["class_name"], "car")
        self.assertEqual(res["vehicle_recommendation"]["vehicle_id"], "V_EICHER_19FT")

    # -------------------------------------------------------------------------
    # 27. Single-Image Multi-Object: Single CAR Image in Extract-All Mode
    # -------------------------------------------------------------------------
    def test_single_image_multi_object_on_single_car_image(self):
        """Verify extract_all_objects=True on an image with 1 car produces 1-item multi-object breakdown."""
        car_img_path = os.path.join(PROJECT_ROOT, "data", "car.jpg")
        if not os.path.exists(car_img_path):
            self.skipTest("data/car.jpg not present on disk.")

        pipeline_yolo = CargoAnalysisPipeline(detector_backend="auto")
        res = pipeline_yolo.analyze(car_img_path, quantity=1, extract_all_objects=True)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["mode"], "SINGLE_IMAGE_MULTI_OBJECT")
        self.assertEqual(len(res["detected_objects"]), 1)
        self.assertEqual(res["detected_objects"][0]["category"], "car")
        self.assertEqual(res["shipment_summary"]["total_weight_kg"], 1400.0)
        self.assertEqual(res["vehicle_recommendation"]["vehicle_id"], "V_EICHER_19FT")

    # -------------------------------------------------------------------------
    # 28. Payload Limit: 50 Boxes x 12 kg = 600 kg Excludes 3-Wheeler Auto
    # -------------------------------------------------------------------------
    def test_fifty_boxes_payload_weight_eliminates_sub_600kg_vehicles(self):
        """Verify 50 boxes * 12 kg = 600 kg rejects 3-Wheeler Auto (500 kg limit) and selects Tata Ace."""
        box_dims = {"length_cm": 45.0, "width_cm": 35.0, "height_cm": 30.0, "weight_kg": 12.0}
        cargo_sum, _ = self.pipeline.calculate_cargo_requirements(box_dims, "box", quantity=50)

        self.assertEqual(cargo_sum["total_weight_kg"], 600.0)
        # Volume for 50 boxes: 50 * 0.04725 / 0.80 = 2.953 m3
        # 3-Wheeler Auto (payload 500 kg) must be eliminated because 600 kg > 500 kg.
        # Tata Ace (payload 850 kg, volume 4.78 m3) must accommodate it.
        rec, _ = self.pipeline.recommend_vehicle(
            category="box",
            dimensions=box_dims,
            cargo_summary=cargo_sum,
            quantity=50,
        )
        self.assertNotEqual(rec["vehicle_id"], "V_3W_AUTO")
        self.assertEqual(rec["vehicle_id"], "V_TATA_ACE")
        self.assertIn("600.0 kg estimated payload", rec["reason"])

    # -------------------------------------------------------------------------
    # 29. Multi-Load: Quantities and Images Length Mismatch Returns Error
    # -------------------------------------------------------------------------
    def test_multiload_mismatched_images_and_quantities_returns_error(self):
        """Verify mismatched lengths between image_paths and quantities returns structured error."""
        res = self.pipeline.analyze_multiple(
            image_paths=[self.valid_image_path, self.valid_image_path],
            quantities=[1],  # 1 quantity for 2 images
        )
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("Length of quantities (1) must match length of image_paths (2)", res["error"])

    # -------------------------------------------------------------------------
    # 30. Empty Shipment Handling
    # -------------------------------------------------------------------------
    def test_empty_shipment_handling_produces_no_fake_recommendation(self):
        """Verify empty image lists or empty item lists return error / clean reason without recommending a vehicle."""
        # 1. analyze_multiple with empty list
        res_empty = self.pipeline.analyze_multiple([])
        self.assertEqual(res_empty["status"], "ERROR")
        self.assertIn("image_paths must be a non-empty list", res_empty["error"])

        # 2. recommend_vehicle_for_shipment with empty items list
        rec, warnings = self.pipeline.recommend_vehicle_for_shipment(items=[], total_volume_m3=0.0, total_floor_area_m2=0.0)
        self.assertIsNone(rec["vehicle_id"])
        self.assertIn("Shipment is empty", rec["reason"])

    # -------------------------------------------------------------------------
    # 31. Constraint: Weight Fits but Floor Area Exceeds Eliminates Vehicle
    # -------------------------------------------------------------------------
    def test_floor_area_constraint_eliminates_underdimensioned_bed(self):
        """Verify floor area exceeding vehicle bed area eliminates vehicle even when weight and volume fit."""
        # 3 items requiring 2.5 m2 bed area, 1.2 m3 volume, 30.0 kg weight
        # 3-Wheeler Auto has 1.88 m2 floor bed area -> eliminated!
        # Tata Ace has 3.19 m2 floor bed area -> selected!
        items = [
            {"category": "chair", "quantity": 3, "dimensions": {"length_cm": 60.0, "width_cm": 60.0, "height_cm": 90.0}},
        ]
        rec, _ = self.pipeline.recommend_vehicle_for_shipment(
            items=items,
            total_volume_m3=1.2,
            total_floor_area_m2=2.50,
            total_weight_kg=30.0,
        )
        self.assertNotEqual(rec["vehicle_id"], "V_3W_AUTO")
        self.assertEqual(rec["vehicle_id"], "V_TATA_ACE")
        self.assertIn("2.50 m² floor bed area", rec["reason"])

    # -------------------------------------------------------------------------
    # 32. Constraint: Height Exceeding Usable Clearance Eliminates Vehicle
    # -------------------------------------------------------------------------
    def test_item_height_exceeding_clearance_eliminates_vehicle(self):
        """Verify an item with height 175 cm eliminates 3-Wheeler Auto (120 cm) and Tata Ace (145 cm)."""
        items = [
            {"category": "refrigerator", "quantity": 1, "dimensions": {"length_cm": 70.0, "width_cm": 70.0, "height_cm": 175.0}},
        ]
        rec, _ = self.pipeline.recommend_vehicle_for_shipment(
            items=items,
            total_volume_m3=1.0,
            total_floor_area_m2=0.49,
            total_weight_kg=75.0,
        )
        # Auto (120 cm) and Tata Ace (145 cm) must be eliminated
        self.assertNotIn(rec["vehicle_id"], ["V_3W_AUTO", "V_TATA_ACE"])
        self.assertEqual(rec["vehicle_id"], "V_BOLERO_PICKUP")

    # -------------------------------------------------------------------------
    # 33. Multi-Image 3-CAR Shipment Aggregation Regression
    # -------------------------------------------------------------------------
    def test_three_car_multi_image_aggregation_regression(self):
        """Verify 3 separate car items aggregate into 35.235 m3, 24.3 m2, 4200 kg and recommend V_CAR_CARRIER_MULTI."""
        car_dims = {"length_cm": 450.0, "width_cm": 180.0, "height_cm": 145.0, "weight_kg": 1400.0}
        car_sum, _ = self.pipeline.calculate_cargo_requirements(car_dims, "car", quantity=1)

        items = [
            {"category": "car", "quantity": 1, "dimensions": car_dims, "cargo_summary": car_sum},
            {"category": "car", "quantity": 1, "dimensions": car_dims, "cargo_summary": car_sum},
            {"category": "car", "quantity": 1, "dimensions": car_dims, "cargo_summary": car_sum},
        ]

        total_vol = round(car_sum["total_volume_m3"] * 3, 4)
        total_area = round(car_sum["required_floor_area_m2"] * 3, 4)
        total_wt = round(car_sum["total_weight_kg"] * 3, 2)

        self.assertEqual(total_vol, 35.235)
        self.assertEqual(total_area, 24.3)
        self.assertEqual(total_wt, 4200.0)

        rec, _ = self.pipeline.recommend_vehicle_for_shipment(
            items=items,
            total_volume_m3=total_vol,
            total_floor_area_m2=total_area,
            total_weight_kg=total_wt,
        )
        self.assertEqual(rec["vehicle_id"], "V_CAR_CARRIER_MULTI")
        self.assertIn("Accommodates combined shipment (3 items:", rec["reason"])

    # -------------------------------------------------------------------------
    # 34. Individual 3D Dimension Fit: Cargo Too Long Eliminates Vehicle
    # -------------------------------------------------------------------------
    def test_cargo_too_long_eliminates_vehicle_despite_volume_and_payload(self):
        """Verify cargo item with length exceeding vehicle length is rejected even if volume and payload fit."""
        # Long cargo: 280 x 40 x 40 cm, 50 kg. Volume = 0.448 / 0.80 = 0.56 m3, floor area = 1.12 m2, payload = 50 kg.
        # Bolero Pickup (length 250 cm, max dim 250 cm, vol 7.43 m3, bed 4.25 m2, payload 1500 kg)
        # 280 cm exceeds Bolero's 250 cm length -> Bolero must be eliminated!
        # Tata 407 (length 420 cm, vol 17.64 m3, payload 3500 kg) accommodates it.
        cargo_dims = {"length_cm": 280.0, "width_cm": 40.0, "height_cm": 40.0, "weight_kg": 50.0}
        cargo_sum, _ = self.pipeline.calculate_cargo_requirements(cargo_dims, "box", quantity=1)

        rec, _ = self.pipeline.recommend_vehicle(
            category="box",
            dimensions=cargo_dims,
            cargo_summary=cargo_sum,
            quantity=1,
        )
        self.assertNotIn(rec["vehicle_id"], ["V_3W_AUTO", "V_TATA_ACE", "V_BOLERO_PICKUP"])
        self.assertEqual(rec["vehicle_id"], "V_TATA_407_14FT")

    # -------------------------------------------------------------------------
    # 35. Individual 3D Dimension Fit: Cargo Too Wide Eliminates Vehicle
    # -------------------------------------------------------------------------
    def test_cargo_too_wide_eliminates_vehicle(self):
        """Verify cargo item whose width exceeds vehicle usable width is rejected."""
        # Bed frame: 200 x 160 x 60 cm, 50 kg.
        # Tata Ace has usable width 145 cm < 160 cm, usable length 220 cm.
        # Lying upright: width 160 > 145; Yaw-rotated: length 200 > 145.
        # Tata Ace must be eliminated due to width constraints!
        # Bolero Pickup (usable width 170 cm, length 250 cm) accommodates it.
        bed_dims = {"length_cm": 200.0, "width_cm": 160.0, "height_cm": 60.0, "weight_kg": 50.0}
        cargo_sum, _ = self.pipeline.calculate_cargo_requirements(bed_dims, "bed", quantity=1)

        rec, _ = self.pipeline.recommend_vehicle(
            category="bed",
            dimensions=bed_dims,
            cargo_summary=cargo_sum,
            quantity=1,
        )
        self.assertNotIn(rec["vehicle_id"], ["V_3W_AUTO", "V_TATA_ACE"])
        self.assertEqual(rec["vehicle_id"], "V_BOLERO_PICKUP")

    # -------------------------------------------------------------------------
    # 36. Individual 3D Dimension Fit: Cargo Too Tall Eliminates Vehicle
    # -------------------------------------------------------------------------
    def test_cargo_too_tall_eliminates_vehicle(self):
        """Verify upright cargo item whose height exceeds vehicle usable height is rejected."""
        # Tall wardrobe/refrigerator: 80 x 80 x 190 cm, 75 kg.
        # Tata Ace height is 150 cm < 190 cm -> eliminated.
        # Bolero Pickup height is 175 cm < 190 cm -> eliminated.
        # Tata 407 (usable height 210 cm) accommodates it.
        tall_dims = {"length_cm": 80.0, "width_cm": 80.0, "height_cm": 190.0, "weight_kg": 75.0}
        cargo_sum, _ = self.pipeline.calculate_cargo_requirements(tall_dims, "refrigerator", quantity=1)

        rec, _ = self.pipeline.recommend_vehicle(
            category="refrigerator",
            dimensions=tall_dims,
            cargo_summary=cargo_sum,
            quantity=1,
        )
        self.assertNotIn(rec["vehicle_id"], ["V_3W_AUTO", "V_TATA_ACE", "V_BOLERO_PICKUP"])
        self.assertEqual(rec["vehicle_id"], "V_TATA_407_14FT")

    # -------------------------------------------------------------------------
    # 37. Individual 3D Dimension Fit: Rotation Allows Fit
    # -------------------------------------------------------------------------
    def test_rotation_allows_fit_when_original_orientation_exceeds(self):
        """Verify cargo item fits when rotated even though original width exceeds vehicle width."""
        # Item: length 120 cm, width 180 cm, height 70 cm, 30 kg.
        # Tata Ace: length 220 cm, width 145 cm, height 150 cm.
        # Original orientation: width 180 > 145 (fails).
        # Rotated orientation: length 180 <= 220, width 120 <= 145, height 70 <= 150 (FITS).
        dims = {"length_cm": 120.0, "width_cm": 180.0, "height_cm": 70.0, "weight_kg": 30.0}

        # Test helper directly:
        tata_ace = self.pipeline.vehicle_db.get_vehicle_by_id("V_TATA_ACE")
        fits_with_rot = self.pipeline._dimensions_fit_vehicle(dims, tata_ace, allow_rotation=True)
        fits_without_rot = self.pipeline._dimensions_fit_vehicle(dims, tata_ace, allow_rotation=False)
        self.assertTrue(fits_with_rot)
        self.assertFalse(fits_without_rot)

        # Test end-to-end recommendation:
        cargo_sum, _ = self.pipeline.calculate_cargo_requirements(dims, "couch", quantity=1)
        rec, _ = self.pipeline.recommend_vehicle(
            category="couch",
            dimensions=dims,
            cargo_summary=cargo_sum,
            quantity=1,
        )
        self.assertEqual(rec["vehicle_id"], "V_TATA_ACE")

    # -------------------------------------------------------------------------
    # 38. Multi-Load: Individual Dimensional Violation Rejects Candidate Vehicle
    # -------------------------------------------------------------------------
    def test_multiload_individual_dimensional_violation_rejects_vehicle(self):
        """Verify multi-load shipment rejects vehicle when aggregate metrics fit but one item's dimensions exceed."""
        # 4 boxes (45 x 35 x 30 cm, 12 kg each) + 1 long beam (300 x 20 x 20 cm, 20 kg).
        # Total volume: 4 * 0.04725 / 0.80 + 0.12 / 0.80 = 0.236 + 0.15 = 0.386 m3.
        # Total floor area: 4 * 0.1575 / 4 + 0.06 = 0.1575 + 0.06 = 0.2175 m2.
        # Total weight: 48 + 20 = 68 kg.
        # Tata Ace (vol 4.78 m3, bed 3.19 m2, payload 850 kg) has plenty of volume/payload,
        # but usable length is 220 cm < 300 cm -> Tata Ace MUST be rejected!
        # Bolero Pickup (length 250 cm) < 300 cm -> MUST be rejected!
        # Tata 407 (length 420 cm) -> ACCEPTED!
        box_dims = {"length_cm": 45.0, "width_cm": 35.0, "height_cm": 30.0, "weight_kg": 12.0}
        beam_dims = {"length_cm": 300.0, "width_cm": 20.0, "height_cm": 20.0, "weight_kg": 20.0}

        items = [
            {"category": "box", "quantity": 4, "dimensions": box_dims},
            {"category": "box", "quantity": 1, "dimensions": beam_dims},
        ]

        rec, _ = self.pipeline.recommend_vehicle_for_shipment(
            items=items,
            total_volume_m3=0.386,
            total_floor_area_m2=0.2175,
            total_weight_kg=68.0,
        )
        self.assertNotIn(rec["vehicle_id"], ["V_3W_AUTO", "V_TATA_ACE", "V_BOLERO_PICKUP"])
        self.assertEqual(rec["vehicle_id"], "V_TATA_407_14FT")

    # -------------------------------------------------------------------------
    # 39. Semantic Compatibility: Car Carrier Accepts Car Cargo
    # -------------------------------------------------------------------------
    def test_car_carrier_accepts_car_cargo(self):
        """Verify specialized car carrier is semantically eligible for single and multi-car cargo."""
        car_dims = {"length_cm": 450.0, "width_cm": 180.0, "height_cm": 145.0, "weight_kg": 1400.0}
        car_sum, _ = self.pipeline.calculate_cargo_requirements(car_dims, "car", quantity=1)

        # Single CAR: V_EICHER_19FT is primary, V_CAR_CARRIER_MULTI is valid alternative
        rec_single, _ = self.pipeline.recommend_vehicle("car", car_dims, car_sum, quantity=1)
        self.assertEqual(rec_single["vehicle_id"], "V_EICHER_19FT")
        self.assertIn("Dedicated Multi-Car Carrier (Double-Deck Trailer)", rec_single["alternatives"])

        # 3 CARs: V_CAR_CARRIER_MULTI is primary
        items = [
            {"category": "car", "quantity": 1, "dimensions": car_dims},
            {"category": "car", "quantity": 1, "dimensions": car_dims},
            {"category": "car", "quantity": 1, "dimensions": car_dims},
        ]
        rec_multi, _ = self.pipeline.recommend_vehicle_for_shipment(
            items=items,
            total_volume_m3=35.235,
            total_floor_area_m2=24.3,
            total_weight_kg=4200.0,
        )
        self.assertEqual(rec_multi["vehicle_id"], "V_CAR_CARRIER_MULTI")

    # -------------------------------------------------------------------------
    # 40. Semantic Compatibility: Car Carrier Rejects Non-Car Single Cargo
    # -------------------------------------------------------------------------
    def test_car_carrier_rejects_non_car_cargo(self):
        """Verify non-car items (refrigerator, box, bed, chair, table) never recommend car carrier."""
        test_cases = [
            ("refrigerator", {"length_cm": 70.0, "width_cm": 70.0, "height_cm": 175.0, "weight_kg": 75.0}),
            ("box", {"length_cm": 50.0, "width_cm": 40.0, "height_cm": 30.0, "weight_kg": 15.0}),
            ("bed", {"length_cm": 200.0, "width_cm": 160.0, "height_cm": 60.0, "weight_kg": 50.0}),
            ("chair", {"length_cm": 60.0, "width_cm": 60.0, "height_cm": 90.0, "weight_kg": 8.5}),
            ("table", {"length_cm": 120.0, "width_cm": 75.0, "height_cm": 75.0, "weight_kg": 25.0}),
        ]

        for cat, dims in test_cases:
            c_sum, _ = self.pipeline.calculate_cargo_requirements(dims, cat, quantity=1)
            rec, _ = self.pipeline.recommend_vehicle(cat, dims, c_sum, quantity=1)
            self.assertNotEqual(
                rec["vehicle_id"],
                "V_CAR_CARRIER_MULTI",
                f"Car carrier was incorrectly recommended as primary for {cat}",
            )
            self.assertNotIn(
                "Dedicated Multi-Car Carrier (Double-Deck Trailer)",
                rec["alternatives"],
                f"Car carrier was incorrectly included as alternative for {cat}",
            )

    # -------------------------------------------------------------------------
    # 41. Semantic Compatibility: Car Carrier Rejects Mixed Non-Car Shipments
    # -------------------------------------------------------------------------
    def test_car_carrier_rejects_mixed_non_car_shipments(self):
        """Verify multi-load shipments with only non-car items strictly reject V_CAR_CARRIER_MULTI."""
        refrig_dims = {"length_cm": 70.0, "width_cm": 70.0, "height_cm": 175.0, "weight_kg": 75.0}
        box_dims = {"length_cm": 45.0, "width_cm": 35.0, "height_cm": 30.0, "weight_kg": 12.0}
        bed_dims = {"length_cm": 200.0, "width_cm": 160.0, "height_cm": 60.0, "weight_kg": 50.0}
        chair_dims = {"length_cm": 60.0, "width_cm": 60.0, "height_cm": 90.0, "weight_kg": 8.5}

        # Shipment 1: Refrigerator + Box + Bed
        shipment_1 = [
            {"category": "refrigerator", "quantity": 1, "dimensions": refrig_dims},
            {"category": "box", "quantity": 4, "dimensions": box_dims},
            {"category": "bed", "quantity": 1, "dimensions": bed_dims},
        ]
        rec1, _ = self.pipeline.recommend_vehicle_for_shipment(
            items=shipment_1,
            total_volume_m3=4.5,
            total_floor_area_m2=4.0,
            total_weight_kg=173.0,
        )
        self.assertNotEqual(rec1["vehicle_id"], "V_CAR_CARRIER_MULTI")
        self.assertNotIn("Dedicated Multi-Car Carrier (Double-Deck Trailer)", rec1["alternatives"])

        # Shipment 2: Refrigerator + Bed + Chair
        shipment_2 = [
            {"category": "refrigerator", "quantity": 1, "dimensions": refrig_dims},
            {"category": "bed", "quantity": 2, "dimensions": bed_dims},
            {"category": "chair", "quantity": 4, "dimensions": chair_dims},
        ]
        rec2, _ = self.pipeline.recommend_vehicle_for_shipment(
            items=shipment_2,
            total_volume_m3=8.4883,
            total_floor_area_m2=7.61,
            total_weight_kg=209.0,
        )
        self.assertEqual(rec2["vehicle_id"], "V_TATA_407_14FT")
        self.assertNotIn("Dedicated Multi-Car Carrier (Double-Deck Trailer)", rec2["alternatives"])

    # -------------------------------------------------------------------------
    # 42. Semantic Compatibility: Generic Freight Vehicle Selected When Fitting
    # -------------------------------------------------------------------------
    def test_generic_freight_vehicle_selected_for_fitting_non_car_cargo(self):
        """Verify generic freight trucks (e.g. Tata 407, Eicher 19ft) are selected when non-car cargo fits."""
        couch_dims = {"length_cm": 210.0, "width_cm": 90.0, "height_cm": 85.0, "weight_kg": 70.0}
        cargo_sum, _ = self.pipeline.calculate_cargo_requirements(couch_dims, "couch", quantity=5)
        rec, _ = self.pipeline.recommend_vehicle("couch", couch_dims, cargo_sum, quantity=5)
        self.assertIn(rec["vehicle_id"], ["V_TATA_407_14FT", "V_EICHER_19FT"])
        self.assertNotIn("Dedicated Multi-Car Carrier (Double-Deck Trailer)", rec["alternatives"])

    # -------------------------------------------------------------------------
    # 43. Semantic Compatibility: Cannot Be Overridden By Volume/Payload/Dimensions
    # -------------------------------------------------------------------------
    def test_semantic_incompatibility_cannot_be_overridden_by_volume_or_payload(self):
        """Verify huge non-car cargo exceeding single standard vehicle capacity falls back to Eicher 19ft, not Car Carrier."""
        refrig_dims = {"length_cm": 70.0, "width_cm": 70.0, "height_cm": 175.0, "weight_kg": 75.0}
        cargo_sum, _ = self.pipeline.calculate_cargo_requirements(refrig_dims, "refrigerator", quantity=50)

        rec, _ = self.pipeline.recommend_vehicle("refrigerator", refrig_dims, cargo_sum, quantity=50)
        self.assertEqual(rec["vehicle_id"], "V_EICHER_19FT")
        self.assertIn("exceeds standard single-vehicle capacity", rec["reason"])
        self.assertIn("using 19-ft Medium Freight Truck", rec["reason"])
        self.assertNotIn("Dedicated Multi-Car Carrier", rec["reason"])

    # -------------------------------------------------------------------------
    # 44. BUG #3 Regression: Box Image Detection and Classification
    # -------------------------------------------------------------------------
    def test_box_image_detection_and_classification_regression(self):
        """Verify real box image (data/box8.jpg) is correctly classified as box (not chair) and recommends light vehicle."""
        box_img_path = os.path.join(PROJECT_ROOT, "data", "box8.jpg")
        if not os.path.exists(box_img_path):
            self.skipTest("data/box8.jpg not found in workspace")

        res = self.pipeline.analyze(box_img_path, quantity=8)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["classification"]["class_name"], "box")
        self.assertNotEqual(res["classification"]["class_name"], "chair")
        self.assertGreaterEqual(res["classification"]["confidence"], 0.50)

        # 8 boxes (45x35x30 cm, 12 kg each = 96 kg payload) should easily fit in 3-Wheeler Auto
        rec = res["vehicle_recommendation"]
        self.assertEqual(rec["vehicle_id"], "V_3W_AUTO")
        self.assertIn("box", rec["reason"].lower())

    # -------------------------------------------------------------------------
    # 45. BUG #3 Regression: High-Confidence Genuine Chair Remains Chair
    # -------------------------------------------------------------------------
    def test_genuine_chair_detection_remains_chair(self):
        """Verify genuine chair image is correctly detected and classified as chair."""
        chair_img_path = os.path.join(
            PROJECT_ROOT,
            "deployment_dataset_expanded",
            "test",
            "images",
            "orig_train_Chair_16_JPG.rf.0261cecaf9b0251cdfb8d7cdf50d7988.jpg",
        )
        if not os.path.exists(chair_img_path):
            self.skipTest("Chair test image not found in workspace")

        res = self.pipeline.analyze(chair_img_path, quantity=1)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["classification"]["class_name"], "chair")
        self.assertNotEqual(res["classification"]["class_name"], "box")

    # -------------------------------------------------------------------------
    # 46. BUG #3 Regression: Car Image Detection Preservation
    # -------------------------------------------------------------------------
    def test_car_detection_preservation(self):
        """Verify car image (data/car.jpg) is correctly detected as car and dispatched to Eicher 19ft."""
        car_img_path = os.path.join(PROJECT_ROOT, "data", "car.jpg")
        if not os.path.exists(car_img_path):
            self.skipTest("data/car.jpg not found in workspace")

        res = self.pipeline.analyze(car_img_path, quantity=1)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["classification"]["class_name"], "car")
        self.assertEqual(res["vehicle_recommendation"]["vehicle_id"], "V_EICHER_19FT")

    # -------------------------------------------------------------------------
    # 47. BUG #3 Regression: Refrigerator Image Detection Preservation
    # -------------------------------------------------------------------------
    def test_refrigerator_detection_preservation(self):
        """Verify refrigerator image (data/refrigeratorx10.jpg) is correctly detected as refrigerator."""
        refrig_img_path = os.path.join(PROJECT_ROOT, "data", "refrigeratorx10.jpg")
        if not os.path.exists(refrig_img_path):
            self.skipTest("data/refrigeratorx10.jpg not found in workspace")

        res = self.pipeline.analyze(refrig_img_path, quantity=1)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["classification"]["class_name"], "refrigerator")

    # -------------------------------------------------------------------------
    # 48. BUG #4 Regression: High-Confidence Detections Are Marked RELIABLE
    # -------------------------------------------------------------------------
    def test_high_confidence_detections_marked_reliable(self):
        """Verify high-confidence Box, Car, and Refrigerator detections receive RELIABLE status."""
        # Box test
        box_path = os.path.join(PROJECT_ROOT, "data", "box8.jpg")
        if os.path.exists(box_path):
            res_box = self.pipeline.analyze(box_path, quantity=1)
            self.assertEqual(res_box["status"], "SUCCESS")
            self.assertEqual(res_box["reliability"], "RELIABLE")
            self.assertGreaterEqual(res_box["classification"]["confidence"], 0.50)

        # Car test
        car_path = os.path.join(PROJECT_ROOT, "data", "car.jpg")
        if os.path.exists(car_path):
            res_car = self.pipeline.analyze(car_path, quantity=1)
            self.assertEqual(res_car["status"], "SUCCESS")
            self.assertEqual(res_car["reliability"], "RELIABLE")
            self.assertGreaterEqual(res_car["classification"]["confidence"], 0.50)

        # Refrigerator test
        refrig_path = os.path.join(PROJECT_ROOT, "data", "refrigeratorx10.jpg")
        if os.path.exists(refrig_path):
            res_ref = self.pipeline.analyze(refrig_path, quantity=1)
            self.assertEqual(res_ref["status"], "SUCCESS")
            self.assertEqual(res_ref["reliability"], "RELIABLE")
            self.assertGreaterEqual(res_ref["classification"]["confidence"], 0.50)

    # -------------------------------------------------------------------------
    # 49. BUG #4 Regression: Low-Confidence Detections Marked REVIEW_REQUIRED
    # -------------------------------------------------------------------------
    def test_low_confidence_detection_marked_review_required(self):
        """Verify detections below 50% confidence receive REVIEW_REQUIRED and generate explicit warning."""
        chair_path = os.path.join(
            PROJECT_ROOT,
            "deployment_dataset_expanded",
            "test",
            "images",
            "orig_train_Chair_16_JPG.rf.0261cecaf9b0251cdfb8d7cdf50d7988.jpg",
        )
        if not os.path.exists(chair_path):
            self.skipTest("Chair test image not found in workspace")

        res = self.pipeline.analyze(chair_path, quantity=1)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertLess(res["classification"]["confidence"], 0.50)
        self.assertEqual(res["reliability"], "REVIEW_REQUIRED")
        self.assertEqual(res["classification"]["reliability"], "REVIEW_REQUIRED")

        # Explicit warning must be present
        warnings_joined = " ".join(res["warnings"])
        self.assertIn("below reliable threshold", warnings_joined)
        self.assertIn("Manual verification recommended", warnings_joined)

    # -------------------------------------------------------------------------
    # 50. BUG #4 Regression: Multi-Object Low-Confidence Objects Handled Safely
    # -------------------------------------------------------------------------
    def test_multi_object_low_confidence_table_flagged_for_review(self):
        """Verify multi-object extraction tags low-confidence table as REVIEW_REQUIRED while preserving reliable objects."""
        multi_img = os.path.join(PROJECT_ROOT, "deployment_dataset_expanded", "test", "images", "coco_000000057238.jpg")
        if not os.path.exists(multi_img):
            self.skipTest("coco_000000057238.jpg not found in workspace")

        res = self.pipeline.analyze(multi_img, extract_all_objects=True)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["mode"], "SINGLE_IMAGE_MULTI_OBJECT")

        detected_map = {item["category"]: item for item in res["detected_objects"]}
        self.assertIn("refrigerator", detected_map)
        self.assertIn("chair", detected_map)
        self.assertIn("table", detected_map)

        # High-confidence items
        self.assertEqual(detected_map["refrigerator"]["reliability"], "RELIABLE")
        self.assertTrue(detected_map["refrigerator"]["is_dispatch_ready"])
        self.assertEqual(detected_map["chair"]["reliability"], "RELIABLE")
        self.assertTrue(detected_map["chair"]["is_dispatch_ready"])

        # Low-confidence item (~29% Table)
        self.assertEqual(detected_map["table"]["reliability"], "REVIEW_REQUIRED")
        self.assertFalse(detected_map["table"]["is_dispatch_ready"])

        # Reliability breakdown in shipment summary
        breakdown = res["shipment_summary"]["reliability_breakdown"]
        self.assertEqual(breakdown["reliable_items_count"], 3)  # 1 refrig + 2 chairs
        self.assertEqual(breakdown["review_required_items_count"], 1)  # 1 table
        self.assertIn("table", breakdown["review_required_categories"])

        # Warnings include explicit review notification
        warnings_str = " ".join(res["warnings"])
        self.assertIn("Low-confidence cargo detected: 'table'", warnings_str)
        self.assertIn("marked REVIEW_REQUIRED", warnings_str)

    # -------------------------------------------------------------------------
    # 51. BUG #6 Physical Dimension Estimation: ArUco with Aspect Depth Inference
    # -------------------------------------------------------------------------
    def test_aruco_analysis_with_aspect_depth_inference(self):
        """Verify end-to-end pipeline analysis with infer_aspect_depth=True calculates volumetric metrics."""
        res = self.pipeline.analyze(
            image_path=self.marker_image_path,
            quantity=1,
            known_marker_size_cm=10.0,
            object_bbox_px=(200, 200, 400, 300),
            infer_aspect_depth=True,
        )
        self.assertEqual(res["status"], "SUCCESS")
        dim_res = res["dimensions"]

        self.assertEqual(dim_res["status"], "MEASURED")
        self.assertIn(dim_res["source"], ["aruco_plus_aspect_prior", "aruco_reference"])
        self.assertIsNotNone(dim_res["length_cm"])
        self.assertIsNotNone(dim_res["height_cm"])

        # Estimation metadata and uncertainty bounds are present
        self.assertIn("estimation_method", dim_res)
        self.assertIn("uncertainty_cm", dim_res)
        self.assertIn("uncertainty_percent", dim_res)

        # If category is known and depth is inferred, volumetric requirements are computed
        if dim_res["width_cm"] is not None:
            self.assertEqual(dim_res["estimation_method"], "ARUCO_PLUS_ASPECT_PRIOR")
            self.assertIsNotNone(res["cargo_summary"]["total_volume_m3"])
            self.assertIsNotNone(res["cargo_summary"]["required_floor_area_m2"])
            self.assertIsNotNone(res["vehicle_recommendation"]["vehicle_id"])


if __name__ == "__main__":
    unittest.main()
