"""
test_openimages_supplement.py - Unit Tests for Open Images V7 Supplemental Ingestion
===================================================================================
Project: Cargo Vision Logistics System
Purpose: Independently tests the acquisition, validation, deduplication, and staging
         of supplemental Open Images V7 dataset samples.
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

from scripts.acquire_openimages_supplement import (
    CANONICAL_24_CLASSES,
    CLASS_NAME_TO_ID,
    OPENIMAGES_MID_TO_CLASS,
    validate_yolo_bbox,
    compute_image_sha256,
    OpenImagesSupplementAcquirer,
)


class TestOpenImagesSupplement(unittest.TestCase):
    """Test suite for Open Images V7 supplemental acquisition."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.supplement_dir = os.path.join(self.test_dir, "test_supplement")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_24_class_openimages_mid_coverage(self):
        """Verify all 24 canonical classes have valid mapped MIDs in Open Images V7."""
        mapped_classes = set(c[0] for c in OPENIMAGES_MID_TO_CLASS.values())
        for cname in CANONICAL_24_CLASSES:
            self.assertIn(cname, mapped_classes, f"Missing OpenImages MID mapping for '{cname}'")

    def test_yolo_bbox_coordinate_validation(self):
        """Verify bounding box validator rejects malformed coordinates."""
        # Valid box
        is_val, msg = validate_yolo_bbox(0, 0.5, 0.5, 0.2, 0.3, nc=24)
        self.assertTrue(is_val, msg)

        # Negative coordinate
        is_val, _ = validate_yolo_bbox(0, -0.1, 0.5, 0.2, 0.3, nc=24)
        self.assertFalse(is_val)

        # Out of bounds (> 1.0)
        is_val, _ = validate_yolo_bbox(0, 0.5, 1.1, 0.2, 0.3, nc=24)
        self.assertFalse(is_val)

        # Zero dimension
        is_val, _ = validate_yolo_bbox(0, 0.5, 0.5, 0.0, 0.3, nc=24)
        self.assertFalse(is_val)

    def test_sha256_image_deduplication(self):
        """Verify SHA-256 hash calculation reliably flags identical images."""
        p1 = os.path.join(self.test_dir, "img1.png")
        p2 = os.path.join(self.test_dir, "img2.png")
        p3 = os.path.join(self.test_dir, "img3.png")

        im1 = Image.new("RGB", (32, 32), color="green")
        im1.save(p1)
        im1.save(p2)

        im2 = Image.new("RGB", (32, 32), color="yellow")
        im2.save(p3)

        h1 = compute_image_sha256(p1)
        h2 = compute_image_sha256(p2)
        h3 = compute_image_sha256(p3)

        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)

    def test_existing_dataset_indexing_prevents_duplicate_acquisition(self):
        """Verify existing hashes and IDs prevent duplicate ingestion."""
        acquirer = OpenImagesSupplementAcquirer(
            supplement_dir=self.supplement_dir,
            cache_dir="data/public_cache",
        )
        acquirer.scan_existing_datasets_to_prevent_duplicates()
        self.assertTrue(len(acquirer.existing_hashes) > 0)
        self.assertTrue(len(acquirer.existing_image_ids) > 0)


if __name__ == "__main__":
    unittest.main()
