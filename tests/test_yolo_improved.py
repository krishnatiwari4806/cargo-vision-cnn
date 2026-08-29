"""
test_yolo_improved.py - Unit Tests for Enhanced YOLO Pipeline & Architecture
============================================================================
Project: Cargo Vision Logistics System
Purpose: Verifies dataset integrity, 24-class structure, hyperparameter boundaries,
         prediction schema, bounding box coordinate validity, confidence ranges,
         evaluation reporting format, and error handling.
"""

import os
import sys
import unittest
import yaml
import tempfile
import json
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.predict_yolo import YOLOCargoDetector
from scripts.evaluate_yolo_improved import evaluate_improved_detector

EXPECTED_24_CLASSES = [
    "Bed", "Box", "Chair", "Couch", "Desk", "Door", "Drawer", "Laundry",
    "Person", "Shoe", "Sink", "Suitcase", "Table", "Tv", "book shelf",
    "cat", "dog", "fan", "mirror", "refrigerator", "stool", "stove",
    "toilet", "trashcan"
]


class TestYOLOImprovedPipeline(unittest.TestCase):
    """
    Unit test suite for enhanced YOLO training, evaluation, and inference validation.
    """

    @classmethod
    def setUpClass(cls):
        cls.data_yaml_path = os.path.join(PROJECT_ROOT, "deployment_dataset", "data.yaml")
        cls.temp_dir = tempfile.mkdtemp()
        cls.synthetic_img_path = os.path.join(cls.temp_dir, "test_synth_cargo.jpg")

        # Create valid synthetic image
        img = Image.new("RGB", (640, 640), color=(100, 120, 140))
        img.save(cls.synthetic_img_path, "JPEG")

        # Detector instance
        cls.detector = YOLOCargoDetector(
            model_path=os.path.join(PROJECT_ROOT, "models", "cargo_yolo_24class_best.pt"),
            conf_threshold=0.10,
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    # -------------------------------------------------------------------------
    # 1. Dataset Configuration & Class Integrity
    # -------------------------------------------------------------------------
    def test_dataset_yaml_structure_and_classes(self):
        """Verify data.yaml exists, defines exactly 24 classes matching specifications."""
        self.assertTrue(os.path.exists(self.data_yaml_path))
        with open(self.data_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertIn("nc", data)
        self.assertEqual(data["nc"], 24)
        self.assertIn("names", data)
        self.assertEqual(len(data["names"]), 24)
        self.assertEqual(data["names"], EXPECTED_24_CLASSES)

    # -------------------------------------------------------------------------
    # 2. Prediction Schema & Field Formatting
    # -------------------------------------------------------------------------
    def test_prediction_schema_completeness(self):
        """Verify detector inference produces required JSON structure."""
        result = self.detector.predict(self.synthetic_img_path)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("image_path", result)
        self.assertIn("image_dimensions_px", result)
        self.assertIn("detected_objects_count", result)
        self.assertIn("primary_detection", result)
        self.assertIn("detections", result)
        self.assertIsInstance(result["detections"], list)

    # -------------------------------------------------------------------------
    # 3. Bounding Box Validity & Normalization Bounds
    # -------------------------------------------------------------------------
    def test_detection_boxes_geometry_validity(self):
        """Verify detected bounding boxes have valid positive dimensions and coordinates."""
        test_images_dir = os.path.join(PROJECT_ROOT, "deployment_dataset", "test", "images")
        if os.path.exists(test_images_dir):
            test_files = [f for f in os.listdir(test_images_dir) if f.endswith((".jpg", ".png"))]
            if test_files:
                sample_path = os.path.join(test_images_dir, test_files[0])
                result = self.detector.predict(sample_path)
                self.assertEqual(result["status"], "SUCCESS")

                for det in result["detections"]:
                    # Pixel bbox
                    px_box = det["bbox_xyxy_px"]
                    self.assertEqual(len(px_box), 4)
                    x1, y1, x2, y2 = px_box
                    self.assertGreaterEqual(x2, x1)
                    self.assertGreaterEqual(y2, y1)

                    # Normalized bbox
                    if det["bbox_normalized"] is not None:
                        nx1, ny1, nx2, ny2 = det["bbox_normalized"]
                        self.assertTrue(0.0 <= nx1 <= 1.0)
                        self.assertTrue(0.0 <= ny1 <= 1.0)
                        self.assertTrue(0.0 <= nx2 <= 1.0)
                        self.assertTrue(0.0 <= ny2 <= 1.0)

    # -------------------------------------------------------------------------
    # 4. Confidence Value Range Validity
    # -------------------------------------------------------------------------
    def test_detection_confidence_range(self):
        """Verify all detection confidence scores strictly lie in [0.0, 1.0]."""
        test_images_dir = os.path.join(PROJECT_ROOT, "deployment_dataset", "test", "images")
        if os.path.exists(test_images_dir):
            for f in os.listdir(test_images_dir)[:3]:
                if f.endswith((".jpg", ".png")):
                    res = self.detector.predict(os.path.join(test_images_dir, f))
                    for det in res["detections"]:
                        self.assertGreaterEqual(det["confidence"], 0.0)
                        self.assertLessEqual(det["confidence"], 1.0)

    # -------------------------------------------------------------------------
    # 5. Missing & Corrupt Image Error Handling
    # -------------------------------------------------------------------------
    def test_missing_image_returns_error_cleanly(self):
        """Verify non-existent image paths return structured ERROR without crashing."""
        res = self.detector.predict("non_existent_file_path_xyz_123.jpg")
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("error", res)
        self.assertEqual(res["detected_objects_count"], 0)
        self.assertIsNone(res["primary_detection"])

    # -------------------------------------------------------------------------
    # 6. Evaluation Script Error Handling on Invalid Dataset/Model
    # -------------------------------------------------------------------------
    def test_evaluator_missing_model_raises_filenotfound(self):
        """Verify evaluate_improved_detector raises FileNotFoundError when model path is invalid."""
        with self.assertRaises(FileNotFoundError):
            evaluate_improved_detector(model_path="non_existent_model_weights.pt")

    def test_evaluator_missing_yaml_raises_filenotfound(self):
        """Verify evaluate_improved_detector raises FileNotFoundError when data_yaml is missing."""
        # Use existing model if available
        model_p = os.path.join(PROJECT_ROOT, "models", "cargo_yolo_24class_best.pt")
        if not os.path.exists(model_p):
            model_p = "yolov8n.pt"
        with self.assertRaises(FileNotFoundError):
            evaluate_improved_detector(model_path=model_p, data_yaml="non_existent_data.yaml")

    # -------------------------------------------------------------------------
    # 7. Evaluation Output Schema Integrity
    # -------------------------------------------------------------------------
    def test_evaluation_json_schema_structure(self):
        """Verify evaluation report schema adheres to required JSON format."""
        temp_out_json = os.path.join(self.temp_dir, "test_eval_report.json")
        
        # Test evaluation structure on valid model & yaml
        model_p = os.path.join(PROJECT_ROOT, "models", "cargo_yolo_24class_best.pt")
        if not os.path.exists(model_p):
            model_p = "yolov8n.pt"

        report = evaluate_improved_detector(
            model_path=model_p,
            data_yaml=self.data_yaml_path,
            split="test",
            imgsz=320,
            batch_size=8,
            output_json=temp_out_json,
        )

        self.assertIn("overall_metrics", report)
        self.assertIn("per_class_metrics", report)
        self.assertIn("test_images_count", report)
        self.assertIn("ground_truth_objects_count", report)
        self.assertEqual(report["classes_count"], 24)

        for c_res in report["per_class_metrics"]:
            self.assertIn("class_name", c_res)
            self.assertIn("ground_truth_instances", c_res)
            self.assertIn("precision", c_res)
            self.assertIn("recall", c_res)
            self.assertIn("mAP50", c_res)
            self.assertIn("mAP50_95", c_res)


if __name__ == "__main__":
    unittest.main()
