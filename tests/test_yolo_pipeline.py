"""
test_yolo_pipeline.py - Unit Tests for YOLO 24-Class Object Detection Pipeline
==============================================================================
Project: Cargo Vision Logistics System
Purpose: Tests data.yaml schema, 24-class configuration, detector instantiation,
         prediction schema, bounding box validity, and evaluation reporting.
"""

import os
import sys
import unittest
import yaml
import tempfile
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.predict_yolo import YOLOCargoDetector

EXPECTED_24_CLASSES = [
    "Bed", "Box", "Chair", "Couch", "Desk", "Door", "Drawer", "Laundry",
    "Person", "Shoe", "Sink", "Suitcase", "Table", "Tv", "book shelf",
    "cat", "dog", "fan", "mirror", "refrigerator", "stool", "stove",
    "toilet", "trashcan"
]


class TestYOLOPipeline(unittest.TestCase):
    """
    Unit test suite for YOLO detection configuration, detector inference, and schema.
    """

    @classmethod
    def setUpClass(cls):
        cls.data_yaml_path = os.path.join(PROJECT_ROOT, "deployment_dataset", "data.yaml")
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_image_path = os.path.join(cls.temp_dir, "test_synth_scene.jpg")

        # Create a synthetic test image
        img = Image.new("RGB", (320, 240), color=(120, 140, 160))
        img.save(cls.test_image_path, "JPEG")

        # Instantiate detector (uses fallback yolov8n.pt if cargo_yolo_24class_best.pt not yet trained)
        cls.detector = YOLOCargoDetector(conf_threshold=0.10)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    # -------------------------------------------------------------------------
    # 1. data.yaml Loading and Verification
    # -------------------------------------------------------------------------
    def test_data_yaml_exists_and_loads(self):
        """Verify data.yaml exists and is valid YAML."""
        self.assertTrue(os.path.exists(self.data_yaml_path), f"Missing data.yaml at {self.data_yaml_path}")
        with open(self.data_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertIsInstance(data, dict)
        self.assertIn("nc", data)
        self.assertIn("names", data)

    # -------------------------------------------------------------------------
    # 2. Class Count = 24 and Class Names
    # -------------------------------------------------------------------------
    def test_class_count_equals_24_and_matches_specs(self):
        """Verify nc == 24 and class names match all 24 required categories."""
        with open(self.data_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertEqual(data["nc"], 24)
        self.assertEqual(len(data["names"]), 24)
        for expected_cls in EXPECTED_24_CLASSES:
            self.assertIn(expected_cls, data["names"], f"Missing expected class: {expected_cls}")

    # -------------------------------------------------------------------------
    # 3. Image Path Validation & Error Handling
    # -------------------------------------------------------------------------
    def test_invalid_image_path_returns_error_schema(self):
        """Verify non-existent image path returns structured ERROR dictionary."""
        res = self.detector.predict("non_existent_yolo_test_file_999.jpg")
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("error", res)
        self.assertEqual(res["detected_objects_count"], 0)
        self.assertIsNone(res["primary_detection"])
        self.assertEqual(len(res["detections"]), 0)

    # -------------------------------------------------------------------------
    # 4. Prediction Output Schema
    # -------------------------------------------------------------------------
    def test_prediction_output_schema_integrity(self):
        """Verify detector output contains all required fields."""
        res = self.detector.predict(self.test_image_path)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("image_path", res)
        self.assertIn("detected_objects_count", res)
        self.assertIn("primary_detection", res)
        self.assertIn("detections", res)
        self.assertIsInstance(res["detections"], list)

    # -------------------------------------------------------------------------
    # 5. Bounding Box Validity & Confidence Range
    # -------------------------------------------------------------------------
    def test_detection_bbox_and_confidence_validity(self):
        """Verify detected bounding boxes and confidences are well-formed."""
        # Test on one real dataset image if available
        test_img_real = os.path.join(PROJECT_ROOT, "deployment_dataset", "test", "images")
        if os.path.exists(test_img_real):
            sample_files = [f for f in os.listdir(test_img_real) if f.endswith((".jpg", ".png"))]
            if sample_files:
                sample_img_path = os.path.join(test_img_real, sample_files[0])
                res = self.detector.predict(sample_img_path)
                self.assertEqual(res["status"], "SUCCESS")

                for det in res["detections"]:
                    # Confidence check
                    self.assertGreaterEqual(det["confidence"], 0.0)
                    self.assertLessEqual(det["confidence"], 1.0)

                    # Bounding box check
                    bb = det["bbox_xyxy_px"]
                    self.assertEqual(len(bb), 4)
                    x1, y1, x2, y2 = bb
                    self.assertGreaterEqual(x2, x1)
                    self.assertGreaterEqual(y2, y1)

    # -------------------------------------------------------------------------
    # 6. Multiple Object Handling and Confidence Sorting
    # -------------------------------------------------------------------------
    def test_detections_sorted_by_confidence(self):
        """Verify detections list is sorted in descending order of confidence."""
        test_img_real = os.path.join(PROJECT_ROOT, "deployment_dataset", "train", "images")
        if os.path.exists(test_img_real):
            sample_files = [f for f in os.listdir(test_img_real) if f.endswith((".jpg", ".png"))]
            if sample_files:
                sample_img_path = os.path.join(test_img_real, sample_files[0])
                res = self.detector.predict(sample_img_path)
                if len(res["detections"]) >= 2:
                    confs = [d["confidence"] for d in res["detections"]]
                    self.assertEqual(confs, sorted(confs, reverse=True))


if __name__ == "__main__":
    unittest.main()
