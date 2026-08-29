"""
test_dataset_expansion.py - Unit Tests for Dataset Expansion & Ingestion Engine
================================================================================
Project: Cargo Vision Logistics System
Purpose: Unit tests for scripts/expand_dataset.py and scripts/audit_expanded_dataset.py.
         Verifies 24-class taxonomy preservation, YOLO coordinate boundary enforcement,
         image deduplication via SHA-256, stratified split partitioning, and YAML integrity.
"""

import os
import sys
import unittest
import tempfile
import shutil
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.expand_dataset import (
    CANONICAL_24_CLASSES,
    CLASS_NAME_TO_ID,
    validate_yolo_bbox,
    compute_image_sha256,
    normalize_class_name,
    DatasetExpander,
)
from scripts.audit_expanded_dataset import audit_yolo_dataset


class TestDatasetExpansionPipeline(unittest.TestCase):
    """Test suite for dataset expansion and integrity auditing."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_dataset_dir = os.path.join(self.test_dir, "test_expanded_dataset")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_canonical_24_class_taxonomy_completeness(self):
        """Verify CANONICAL_24_CLASSES contains exactly 24 classes matching project specs."""
        self.assertEqual(len(CANONICAL_24_CLASSES), 24)
        expected_classes = [
            "Bed", "Box", "Chair", "Couch", "Desk", "Door", "Drawer", "Laundry",
            "Person", "Shoe", "Sink", "Suitcase", "Table", "Tv", "book shelf",
            "cat", "dog", "fan", "mirror", "refrigerator", "stool", "stove",
            "toilet", "trashcan"
        ]
        self.assertEqual(CANONICAL_24_CLASSES, expected_classes)
        self.assertEqual(len(CLASS_NAME_TO_ID), 24)

    def test_class_normalization_aliases(self):
        """Verify public dataset aliases map to canonical 24 classes correctly."""
        self.assertEqual(normalize_class_name("armchair"), "Chair")
        self.assertEqual(normalize_class_name("sofa"), "Couch")
        self.assertEqual(normalize_class_name("dining table"), "Table")
        self.assertEqual(normalize_class_name("chest of drawers"), "Drawer")
        self.assertEqual(normalize_class_name("fridge"), "refrigerator")
        self.assertEqual(normalize_class_name("waste container"), "trashcan")
        self.assertIsNone(normalize_class_name("airplane_unknown_xyz"))

    def test_yolo_bbox_validation_bounds(self):
        """Verify bounding box validator rejects out-of-bounds, negative, or invalid boxes."""
        # Valid box
        valid, msg = validate_yolo_bbox(2, 0.5, 0.5, 0.4, 0.6, nc=24)
        self.assertTrue(valid, msg)

        # Invalid class index
        valid, msg = validate_yolo_bbox(25, 0.5, 0.5, 0.4, 0.6, nc=24)
        self.assertFalse(valid)
        self.assertIn("out of bounds", msg)

        # Out-of-bounds coordinates
        valid, msg = validate_yolo_bbox(0, 1.2, 0.5, 0.4, 0.4, nc=24)
        self.assertFalse(valid)

        # Non-positive dimension
        valid, msg = validate_yolo_bbox(1, 0.5, 0.5, 0.0, 0.5, nc=24)
        self.assertFalse(valid)
        self.assertIn("Non-positive", msg)

    def test_image_sha256_deduplication(self):
        """Verify SHA-256 correctly identifies exact duplicate image files."""
        img1_path = os.path.join(self.test_dir, "img1.png")
        img2_path = os.path.join(self.test_dir, "img2.png")
        img3_path = os.path.join(self.test_dir, "img3.png")

        # Create identical images
        im1 = Image.new("RGB", (64, 64), color="blue")
        im1.save(img1_path)
        im1.save(img2_path)

        # Create different image
        im2 = Image.new("RGB", (64, 64), color="red")
        im2.save(img3_path)

        hash1 = compute_image_sha256(img1_path)
        hash2 = compute_image_sha256(img2_path)
        hash3 = compute_image_sha256(img3_path)

        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)

    def test_expander_initializes_clean_directory_structure(self):
        """Verify DatasetExpander creates proper train/valid/test directories."""
        expander = DatasetExpander(output_dir=self.output_dataset_dir)
        expander.initialize_output_directory(clean=True)

        for s in ["train", "valid", "test"]:
            self.assertTrue(os.path.isdir(os.path.join(self.output_dataset_dir, s, "images")))
            self.assertTrue(os.path.isdir(os.path.join(self.output_dataset_dir, s, "labels")))

    def test_expander_stratified_split_and_yaml_export(self):
        """Verify expander creates valid split files, copies images, and writes data.yaml."""
        expander = DatasetExpander(output_dir=self.output_dataset_dir)
        expander.initialize_output_directory(clean=True)

        # Create dummy test samples
        for i in range(10):
            img_p = os.path.join(self.test_dir, f"test_box_{i}.png")
            im = Image.new("RGB", (100, 100), color=(i * 20, 50, 100))
            im.save(img_p)
            # Class 1 = Box
            expander.add_sample(img_p, [(1, 0.5, 0.5, 0.6, 0.6)], source_name="synthetic_test")

        counts = expander.build_stratified_split()
        self.assertEqual(sum(counts.values()), 10)
        self.assertTrue(counts["train"] > 0)
        self.assertTrue(counts["valid"] > 0)
        self.assertTrue(counts["test"] > 0)

        # Check data.yaml
        yaml_path = os.path.join(self.output_dataset_dir, "data.yaml")
        self.assertTrue(os.path.exists(yaml_path))

    def test_audit_expanded_dataset_runs_and_produces_report(self):
        """Verify audit_yolo_dataset audits dataset directory and exports JSON."""
        # Run audit on original deployment_dataset
        report = audit_yolo_dataset(
            dataset_dir="deployment_dataset",
            yaml_name="data.yaml",
            output_json=os.path.join(self.test_dir, "audit_output.json"),
        )
        self.assertEqual(report["classes_count"], 24)
        self.assertEqual(report["total_images"], 252)
        self.assertEqual(report["data_integrity"]["cross_split_duplicates_count"], 0)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "audit_output.json")))


if __name__ == "__main__":
    unittest.main()
