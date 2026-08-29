"""
test_dimension_dataset.py - Unit Tests for Dimension Benchmark Dataset Validator
================================================================================
Project: Cargo Vision Logistics System (Phase 3)
Purpose: Tests schema validation, missing field detection, invalid dimension rejection,
         missing image checks, duplicate detection, and malformed JSON resilience.
"""

import os
import sys
import unittest
import json
import tempfile
import shutil

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.validate_dimension_dataset import DimensionDatasetValidator


class TestDimensionDatasetValidator(unittest.TestCase):
    """
    Unit test suite for physical dimension benchmark dataset validation rules.
    """

    def setUp(self):
        self.validator = DimensionDatasetValidator(dataset_dir="data/dimension_benchmark")
        self.valid_sample_dict = {
            "item_id": "BENCH_001",
            "image_filename": "BENCH_001_front.jpg",
            "object_category": "box",
            "image": {
                "width_px": 1920,
                "height_px": 1080,
            },
            "reference": {
                "present": True,
                "type": "aruco_4x4_50",
                "size_cm": 10.0,
            },
            "ground_truth": {
                "length_cm": 45.2,
                "width_cm": 35.0,
                "height_cm": 29.8,
                "weight_kg": 8.4,
            },
            "measurement_method": "measuring_tape",
            "camera": {
                "distance_m": 1.85,
                "view_angle_deg": 0.0,
            },
        }

    # -------------------------------------------------------------------------
    # 1. Valid Annotation
    # -------------------------------------------------------------------------
    def test_valid_annotation_passes_cleanly(self):
        """Verify fully compliant annotation dictionary passes with zero errors."""
        errors = self.validator.validate_annotation_dict(self.valid_sample_dict)
        self.assertEqual(len(errors), 0, f"Expected 0 errors, got: {errors}")

    # -------------------------------------------------------------------------
    # 2. Missing Required Fields
    # -------------------------------------------------------------------------
    def test_missing_required_top_level_fields(self):
        """Verify validator flags missing required fields."""
        required_keys = ["item_id", "image_filename", "object_category", "image", "reference", "ground_truth", "measurement_method"]
        for key in required_keys:
            bad_dict = dict(self.valid_sample_dict)
            del bad_dict[key]
            errors = self.validator.validate_annotation_dict(bad_dict)
            self.assertGreater(len(errors), 0, f"Validator failed to catch missing field: {key}")
            self.assertTrue(any(f"Missing required top-level field: '{key}'" in e for e in errors))

    # -------------------------------------------------------------------------
    # 3. Invalid Dimensions (Negative, Zero, Non-numeric)
    # -------------------------------------------------------------------------
    def test_invalid_ground_truth_dimensions_rejected(self):
        """Verify non-positive dimensions are strictly rejected."""
        # Zero length
        bad_dict = json.loads(json.dumps(self.valid_sample_dict))
        bad_dict["ground_truth"]["length_cm"] = 0.0
        errors = self.validator.validate_annotation_dict(bad_dict)
        self.assertTrue(any("ground_truth.length_cm" in e for e in errors))

        # Negative height
        bad_dict2 = json.loads(json.dumps(self.valid_sample_dict))
        bad_dict2["ground_truth"]["height_cm"] = -15.5
        errors2 = self.validator.validate_annotation_dict(bad_dict2)
        self.assertTrue(any("ground_truth.height_cm" in e for e in errors2))

        # String dimension
        bad_dict3 = json.loads(json.dumps(self.valid_sample_dict))
        bad_dict3["ground_truth"]["width_cm"] = "thirty_five"
        errors3 = self.validator.validate_annotation_dict(bad_dict3)
        self.assertTrue(any("ground_truth.width_cm" in e for e in errors3))

    # -------------------------------------------------------------------------
    # 4. Invalid Reference Marker Size
    # -------------------------------------------------------------------------
    def test_invalid_reference_marker_size(self):
        """Verify invalid marker size when present=True is rejected."""
        bad_dict = json.loads(json.dumps(self.valid_sample_dict))
        bad_dict["reference"]["size_cm"] = -5.0  # Negative size
        errors = self.validator.validate_annotation_dict(bad_dict)
        self.assertTrue(any("reference.size_cm" in e for e in errors))

        bad_dict2 = json.loads(json.dumps(self.valid_sample_dict))
        bad_dict2["reference"]["size_cm"] = 0.0  # Zero size
        errors2 = self.validator.validate_annotation_dict(bad_dict2)
        self.assertTrue(any("reference.size_cm" in e for e in errors2))

    # -------------------------------------------------------------------------
    # 5. Missing Image File Detection
    # -------------------------------------------------------------------------
    def test_missing_image_file_flagged_in_dataset_audit(self):
        """Verify validator flags annotation when corresponding image does not exist on disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            images_dir = os.path.join(tmp_dir, "images")
            annotations_dir = os.path.join(tmp_dir, "annotations")
            os.makedirs(images_dir)
            os.makedirs(annotations_dir)
            os.makedirs(os.path.join(tmp_dir, "metadata"))
            os.makedirs(os.path.join(tmp_dir, "splits"))

            # Write annotation pointing to non-existent image
            annot_path = os.path.join(annotations_dir, "BENCH_001.json")
            with open(annot_path, "w") as f:
                json.dump(self.valid_sample_dict, f)

            temp_validator = DimensionDatasetValidator(dataset_dir=tmp_dir)
            report = temp_validator.validate_dataset()

            self.assertEqual(report["status"], "FAILED")
            self.assertTrue(any("Referenced image does not exist" in e for e in report["errors"]))

    # -------------------------------------------------------------------------
    # 6. Duplicate Image Filenames Detection
    # -------------------------------------------------------------------------
    def test_duplicate_image_filename_flagged(self):
        """Verify validator flags two annotations referencing the same image file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            images_dir = os.path.join(tmp_dir, "images")
            annotations_dir = os.path.join(tmp_dir, "annotations")
            os.makedirs(images_dir)
            os.makedirs(annotations_dir)
            os.makedirs(os.path.join(tmp_dir, "metadata"))
            os.makedirs(os.path.join(tmp_dir, "splits"))

            # Create dummy image
            img_path = os.path.join(images_dir, "BENCH_001_front.jpg")
            with open(img_path, "wb") as f:
                f.write(b"dummy image bytes")

            # Write two annotations referencing the same image
            with open(os.path.join(annotations_dir, "BENCH_001.json"), "w") as f:
                json.dump(self.valid_sample_dict, f)

            sample2 = dict(self.valid_sample_dict)
            sample2["item_id"] = "BENCH_002"
            with open(os.path.join(annotations_dir, "BENCH_002.json"), "w") as f:
                json.dump(sample2, f)

            temp_validator = DimensionDatasetValidator(dataset_dir=tmp_dir)
            report = temp_validator.validate_dataset()

            self.assertEqual(report["status"], "FAILED")
            self.assertTrue(any("Duplicate image_filename referenced" in e for e in report["errors"]))

    # -------------------------------------------------------------------------
    # 7. Malformed JSON File Handling
    # -------------------------------------------------------------------------
    def test_malformed_json_handled_gracefully(self):
        """Verify validator catches and reports syntax errors in JSON files without crashing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            tf.write("{ item_id: 'BENCH_001', unclosed_bracket... ")
            temp_path = tf.name

        try:
            is_valid, errors, data = self.validator.validate_annotation_file(temp_path)
            self.assertFalse(is_valid)
            self.assertIsNone(data)
            self.assertTrue(any("Malformed JSON" in e for e in errors))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # -------------------------------------------------------------------------
    # 8. Clean Initial Benchmark Directory Audit
    # -------------------------------------------------------------------------
    def test_clean_initial_dataset_directory_structure(self):
        """Verify the created data/dimension_benchmark/ directory passes initial structural check."""
        report = self.validator.validate_dataset()
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(len(report["errors"]), 0)
        self.assertEqual(report["total_annotations"], 0)


if __name__ == "__main__":
    unittest.main()
