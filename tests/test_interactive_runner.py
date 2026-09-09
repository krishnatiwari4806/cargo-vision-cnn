"""
test_interactive_runner.py - Unit and Integration Tests for Interactive Cargo Test Runner
========================================================================================
Project: Cargo Vision Logistics System
Purpose: Verifies the functionality, robustness, input validation, and pipeline integration
         of the interactive image test runner (run.py).
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run import (
    find_test_images,
    prompt_image_selection,
    prompt_cargo_category,
    prompt_quantity,
    prompt_user_weight,
    format_analysis_result,
    run_interactive_test,
    CANONICAL_CATEGORIES,
)
from scripts.cargo_analysis_pipeline import CargoAnalysisPipeline


class TestInteractiveRunner(unittest.TestCase):
    """Test suite for run.py interactive runner helpers and pipeline integration."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. No Image Handling
    def test_find_test_images_empty_directory(self):
        """Verify find_test_images returns empty list when directory is empty."""
        images = find_test_images(self.temp_dir)
        self.assertEqual(images, [])

    def test_prompt_image_selection_no_images_returns_none(self):
        """Verify prompt_image_selection returns None and prints instructions when list is empty."""
        result = prompt_image_selection([])
        self.assertIsNone(result)

    # 2. Single Image Auto-Selection
    def test_single_image_auto_selection(self):
        """Verify exactly one image in test directory is auto-selected without prompt."""
        test_img = os.path.join(self.temp_dir, "sample_cargo.jpg")
        with open(test_img, "wb") as f:
            f.write(b"fake_jpg_content")

        images = find_test_images(self.temp_dir)
        self.assertEqual(len(images), 1)
        selected = prompt_image_selection(images)
        self.assertEqual(selected, test_img)

    # 3. Multiple Image Selection
    def test_multiple_image_selection_valid_choice(self):
        """Verify user can select a specific image from multiple candidates."""
        img1 = os.path.join(self.temp_dir, "cargo1.png")
        img2 = os.path.join(self.temp_dir, "cargo2.jpg")
        img3 = os.path.join(self.temp_dir, "cargo3.webp")
        for p in [img1, img2, img3]:
            with open(p, "wb") as f:
                f.write(b"data")

        images = find_test_images(self.temp_dir)
        self.assertEqual(len(images), 3)

        # User selects option '2' (cargo2.jpg)
        mock_input = lambda prompt="": "2"
        selected = prompt_image_selection(images, input_fn=mock_input)
        self.assertEqual(selected, img2)

    def test_multiple_image_selection_default_fallback(self):
        """Verify pressing Enter on multiple image selection selects default option 1."""
        img1 = os.path.join(self.temp_dir, "cargo1.jpg")
        img2 = os.path.join(self.temp_dir, "cargo2.jpg")
        for p in [img1, img2]:
            with open(p, "wb") as f:
                f.write(b"data")

        images = find_test_images(self.temp_dir)
        mock_input = lambda prompt="": ""
        selected = prompt_image_selection(images, input_fn=mock_input)
        self.assertEqual(selected, img1)

    # 4. Quantity Input Validation
    def test_prompt_quantity_valid_and_default(self):
        """Verify valid quantity integer parsing and default value (1)."""
        # Default empty input
        self.assertEqual(prompt_quantity(input_fn=lambda p="": ""), 1)
        # Explicit quantity
        self.assertEqual(prompt_quantity(input_fn=lambda p="": "5"), 5)

    def test_prompt_quantity_invalid_recovery(self):
        """Verify invalid inputs (negative, string) re-prompt until valid."""
        inputs = iter(["abc", "-2", "0", "4"])
        mock_input = lambda p="": next(inputs)
        self.assertEqual(prompt_quantity(input_fn=mock_input), 4)

    # 5. Cargo Category Selection
    def test_prompt_cargo_category_auto_detect(self):
        """Verify option 0 or empty string selects Auto detect (None override)."""
        cat_slug, name = prompt_cargo_category(input_fn=lambda p="": "0")
        self.assertIsNone(cat_slug)
        self.assertEqual(name, "Auto detect")

        cat_slug2, name2 = prompt_cargo_category(input_fn=lambda p="": "")
        self.assertIsNone(cat_slug2)
        self.assertEqual(name2, "Auto detect")

    def test_prompt_cargo_category_manual_declaration(self):
        """Verify selecting 1 selects Box, 2 selects Chair, etc."""
        cat_slug, name = prompt_cargo_category(input_fn=lambda p="": "1")
        self.assertEqual(cat_slug, "box")
        self.assertEqual(name, "Box")

    # 6. Weight Input Validation
    def test_prompt_user_weight_no_weight_known(self):
        """Verify selecting 'No' (2 or default) returns None."""
        self.assertIsNone(prompt_user_weight(input_fn=lambda p="": "2"))
        self.assertIsNone(prompt_user_weight(input_fn=lambda p="": ""))

    def test_prompt_user_weight_yes_with_valid_number(self):
        """Verify selecting 'Yes' (1) and entering a positive number returns float."""
        inputs = iter(["1", "25.5"])
        mock_input = lambda p="": next(inputs)
        wt = prompt_user_weight(input_fn=mock_input)
        self.assertEqual(wt, 25.5)

    def test_prompt_user_weight_invalid_recovery(self):
        """Verify invalid weight (e.g. -5, text) re-prompts until valid positive float."""
        inputs = iter(["1", "invalid", "-10", "0", "18.0"])
        mock_input = lambda p="": next(inputs)
        wt = prompt_user_weight(input_fn=mock_input)
        self.assertEqual(wt, 18.0)

    # 7. Output Formatting
    def test_format_analysis_result_structure(self):
        """Verify format_analysis_result formats structured data with all required sections."""
        dummy_result = {
            "status": "SUCCESS",
            "input_image": "data/test_images/cargo.jpg",
            "classification": {
                "class_name": "box",
                "confidence": 0.88,
                "reliability": "RELIABLE",
            },
            "reliability": "RELIABLE",
            "dimensions": {
                "length_cm": 45.0,
                "width_cm": 35.0,
                "height_cm": 30.0,
                "source": "category_prior",
                "weight_uncertainty_percent": 40.0,
                "weight_uncertainty_kg": 4.8,
            },
            "quantity": 3,
            "cargo_summary": {
                "unit_volume_m3": 0.0473,
                "total_volume_m3": 0.1418,
                "required_floor_area_m2": 0.4725,
                "unit_weight_kg": 12.0,
                "total_weight_kg": 36.0,
                "weight_source": "category_prior",
                "weight_status": "ESTIMATED",
            },
            "vehicle_recommendation": {
                "vehicle_name": "Tata Ace",
                "reason": "Fits cargo cleanly",
                "alternatives": ["Bolero Maxi Truck"],
            },
            "warnings": ["Payload is estimated."],
        }
        report = format_analysis_result(dummy_result, user_declared_cargo=None)
        self.assertIn("CARGO ANALYSIS RESULT", report)
        self.assertIn("Box (Auto-Detected)", report)
        self.assertIn("88.0%", report)
        self.assertIn("Tata Ace", report)
        self.assertIn("±40.0% (±4.8 kg)", report)

    # 8. Pipeline Integration with Real Image (data/box8.jpg)
    def test_pipeline_with_category_override_integration(self):
        """Verify CargoAnalysisPipeline.analyze() honors category_override cleanly."""
        real_img = os.path.join(PROJECT_ROOT, "data", "box8.jpg")
        if not os.path.exists(real_img):
            self.skipTest(f"Test image {real_img} not found.")

        pipeline = CargoAnalysisPipeline()
        # Override with "table"
        result = pipeline.analyze(
            image_path=real_img,
            quantity=2,
            user_weight_kg=30.0,
            category_override="table",
        )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["effective_category"], "table")
        self.assertEqual(result["user_declared_category"], "table")
        self.assertEqual(result["cargo_summary"]["unit_weight_kg"], 30.0)
        self.assertEqual(result["cargo_summary"]["total_weight_kg"], 60.0)
        self.assertEqual(result["cargo_summary"]["weight_source"], "user_provided")

    # 9. Full Interactive Runner Workflow End-to-End Test
    def test_run_interactive_test_end_to_end(self):
        """Verify run_interactive_test executes full flow with simulated user inputs."""
        real_img = os.path.join(PROJECT_ROOT, "data", "box8.jpg")
        if not os.path.exists(real_img):
            self.skipTest(f"Test image {real_img} not found.")

        # Copy image to temp directory
        target_path = os.path.join(self.temp_dir, "box8.jpg")
        shutil.copyfile(real_img, target_path)

        # Simulated inputs:
        # Category: 1 (Box)
        # Quantity: 2
        # Know weight: 1 (Yes)
        # Weight: 15.0
        inputs = iter(["1", "2", "1", "15.0"])
        mock_input = lambda p="": next(inputs)

        exit_code = run_interactive_test(test_dir=self.temp_dir, input_fn=mock_input)
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
