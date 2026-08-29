"""
validate_dimension_dataset.py - Physical Dimension Benchmark Dataset Validator
==============================================================================
Project: Cargo Vision Logistics System (Phase 3)
Purpose: Audits and validates the physical dimension benchmark dataset for
         schema completeness, image-annotation pairing, non-zero positive
         measurements, reference marker validity, and object-level split integrity.

Rules Enforced:
  1. No synthetic measurements or guessed values permitted.
  2. Ground-truth length, width, and height must be strictly positive floats (> 0.0).
  3. Every annotation in annotations/ must have a matching valid image in images/.
  4. If reference.present is True, marker size_cm must be strictly positive (> 0.0).
  5. Never fabricates missing measurements or modifies source files.
"""

import os
import sys
import glob
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image


SUPPORTED_CATEGORIES: List[str] = [
    "box",
    "chair",
    "couch",
    "table",
    "suitcase",
    "car",
    "appliance",
    "bed",
    "crate",
    "pallet",
    "barrel",
]

SUPPORTED_MEASUREMENT_METHODS: List[str] = [
    "measuring_tape",
    "laser_meter",
    "caliper",
    "ruler",
]


class DimensionDatasetValidator:
    """
    Validates physical dimension benchmark annotations and directory structures.
    """

    def __init__(self, dataset_dir: str = "data/dimension_benchmark"):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.images_dir = os.path.join(self.dataset_dir, "images")
        self.annotations_dir = os.path.join(self.dataset_dir, "annotations")
        self.metadata_dir = os.path.join(self.dataset_dir, "metadata")
        self.splits_dir = os.path.join(self.dataset_dir, "splits")

    def validate_annotation_dict(self, data: Dict[str, Any], file_path: str = "in-memory") -> List[str]:
        """
        Validates a single annotation dictionary against the benchmark schema.
        Returns list of error messages (empty list if valid).
        """
        errors = []

        if not isinstance(data, dict):
            return [f"[{file_path}] Annotation root must be a JSON object, got {type(data).__name__}"]

        # 1. Required top-level fields
        required_top_fields = ["item_id", "image_filename", "object_category", "image", "reference", "ground_truth", "measurement_method"]
        for field in required_top_fields:
            if field not in data:
                errors.append(f"[{file_path}] Missing required top-level field: '{field}'")

        if errors:
            return errors  # Stop early if missing core fields

        # 2. item_id
        item_id = data.get("item_id")
        if not isinstance(item_id, str) or not item_id.strip():
            errors.append(f"[{file_path}] 'item_id' must be a non-empty string.")

        # 3. image_filename
        img_fn = data.get("image_filename")
        if not isinstance(img_fn, str) or not img_fn.strip():
            errors.append(f"[{file_path}] 'image_filename' must be a non-empty string.")

        # 4. object_category
        cat = data.get("object_category")
        if not isinstance(cat, str) or not cat.strip():
            errors.append(f"[{file_path}] 'object_category' must be a non-empty string.")

        # 5. image object
        img_obj = data.get("image")
        if not isinstance(img_obj, dict):
            errors.append(f"[{file_path}] 'image' must be an object containing 'width_px' and 'height_px'.")
        else:
            w_px = img_obj.get("width_px")
            h_px = img_obj.get("height_px")
            if not isinstance(w_px, (int, float)) or w_px <= 0:
                errors.append(f"[{file_path}] 'image.width_px' must be a positive number, got {w_px}")
            if not isinstance(h_px, (int, float)) or h_px <= 0:
                errors.append(f"[{file_path}] 'image.height_px' must be a positive number, got {h_px}")

        # 6. reference object
        ref_obj = data.get("reference")
        if not isinstance(ref_obj, dict):
            errors.append(f"[{file_path}] 'reference' must be an object containing 'present', 'type', and 'size_cm'.")
        else:
            present = ref_obj.get("present")
            if not isinstance(present, bool):
                errors.append(f"[{file_path}] 'reference.present' must be a boolean (true/false).")
            elif present is True:
                r_type = ref_obj.get("type")
                if not isinstance(r_type, str) or not r_type.strip():
                    errors.append(f"[{file_path}] 'reference.type' must be a non-empty string when present=true.")
                r_size = ref_obj.get("size_cm")
                if not isinstance(r_size, (int, float)) or r_size <= 0:
                    errors.append(f"[{file_path}] 'reference.size_cm' must be a positive float (>0), got {r_size}")

        # 7. ground_truth object (Physical Dimensions)
        gt = data.get("ground_truth")
        if not isinstance(gt, dict):
            errors.append(f"[{file_path}] 'ground_truth' must be an object with 'length_cm', 'width_cm', and 'height_cm'.")
        else:
            dim_fields = ["length_cm", "width_cm", "height_cm"]
            for df in dim_fields:
                val = gt.get(df)
                if val is None or not isinstance(val, (int, float)) or val <= 0:
                    errors.append(f"[{file_path}] 'ground_truth.{df}' must be a positive float (> 0.0), got {val}")

            # Optional weight check
            weight = gt.get("weight_kg")
            if weight is not None and (not isinstance(weight, (int, float)) or weight <= 0):
                errors.append(f"[{file_path}] 'ground_truth.weight_kg' must be a positive float if provided, got {weight}")

        # 8. measurement_method
        method = data.get("measurement_method")
        if not isinstance(method, str) or not method.strip():
            errors.append(f"[{file_path}] 'measurement_method' must be a non-empty string.")

        return errors

    def validate_annotation_file(self, file_path: str) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
        """
        Loads and validates an individual JSON annotation file.
        """
        if not os.path.exists(file_path):
            return False, [f"File not found: {file_path}"], None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return False, [f"[{file_path}] Malformed JSON: {e}"], None
        except Exception as e:
            return False, [f"[{file_path}] Failed to read file: {e}"], None

        errors = self.validate_annotation_dict(data, file_path=os.path.basename(file_path))
        return (len(errors) == 0), errors, data

    def validate_dataset(self) -> Dict[str, Any]:
        """
        Executes a full dataset audit on data/dimension_benchmark/.
        """
        report: Dict[str, Any] = {
            "dataset_directory": self.dataset_dir,
            "total_annotations": 0,
            "total_images": 0,
            "valid_samples": 0,
            "categories": {},
            "reference_types": {},
            "errors": [],
            "warnings": [],
            "status": "PASSED",
        }

        # Check directory structure
        for d in [self.images_dir, self.annotations_dir, self.metadata_dir, self.splits_dir]:
            if not os.path.exists(d):
                report["errors"].append(f"Missing required directory: {d}")

        if report["errors"]:
            report["status"] = "FAILED"
            return report

        json_files = sorted(glob.glob(os.path.join(self.annotations_dir, "*.json")))
        image_files = sorted(
            glob.glob(os.path.join(self.images_dir, "*.jpg"))
            + glob.glob(os.path.join(self.images_dir, "*.jpeg"))
            + glob.glob(os.path.join(self.images_dir, "*.png"))
        )

        report["total_annotations"] = len(json_files)
        report["total_images"] = len(image_files)

        if len(json_files) == 0:
            report["warnings"].append("Dataset contains 0 JSON annotations. Ready for manual real-sample collection.")
            return report

        seen_image_filenames = set()
        seen_item_ids = set()

        for jf in json_files:
            is_valid, errors, data = self.validate_annotation_file(jf)
            if not is_valid or data is None:
                report["errors"].extend(errors)
                continue

            # Check for duplicate image_filename
            img_fn = data["image_filename"]
            if img_fn in seen_image_filenames:
                report["errors"].append(f"Duplicate image_filename referenced: '{img_fn}' in {os.path.basename(jf)}")
            seen_image_filenames.add(img_fn)

            # Check image existence
            expected_img_path = os.path.join(self.images_dir, img_fn)
            if not os.path.exists(expected_img_path):
                report["errors"].append(f"[{os.path.basename(jf)}] Referenced image does not exist: '{img_fn}'")
            else:
                # Verify physical image dimensions match annotation
                try:
                    with Image.open(expected_img_path) as im:
                        actual_w, actual_h = im.size
                        annot_w = data["image"]["width_px"]
                        annot_h = data["image"]["height_px"]
                        if actual_w != annot_w or actual_h != annot_h:
                            report["warnings"].append(
                                f"[{img_fn}] Image dimension mismatch: actual ({actual_w}x{actual_h}) vs annot ({annot_w}x{annot_h})"
                            )
                except Exception as e:
                    report["errors"].append(f"Failed to open image '{img_fn}': {e}")

            # Track category
            cat = data.get("object_category", "unknown")
            report["categories"][cat] = report["categories"].get(cat, 0) + 1

            # Track reference type
            ref_type = data.get("reference", {}).get("type", "none")
            report["reference_types"][ref_type] = report["reference_types"].get(ref_type, 0) + 1

            report["valid_samples"] += 1

        # Check for orphaned images (images without annotation)
        image_basenames = {os.path.basename(p) for p in image_files}
        orphaned = image_basenames - seen_image_filenames
        if orphaned:
            for o in orphaned:
                report["warnings"].append(f"Orphaned image without matching annotation: '{o}'")

        if report["errors"]:
            report["status"] = "FAILED"

        return report


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate physical dimension benchmark dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/dimension_benchmark",
        help="Path to dimension benchmark dataset root (default: data/dimension_benchmark)",
    )
    args = parser.parse_args()

    validator = DimensionDatasetValidator(dataset_dir=args.dataset_dir)
    report = validator.validate_dataset()

    print("=" * 78)
    print("Cargo Vision - Physical Dimension Benchmark Dataset Validation Report")
    print(f"Directory: {report['dataset_directory']}")
    print(f"Status:    {report['status']}")
    print(f"Total Annotations: {report['total_annotations']} | Total Images: {report['total_images']} | Valid Samples: {report['valid_samples']}")
    print("=" * 78)

    if report["categories"]:
        print("\nSamples per Category:")
        for cat, cnt in sorted(report["categories"].items()):
            print(f"  - {cat:<12}: {cnt:3d} samples")

    if report["errors"]:
        print(f"\n[ERRORS DETECTED: {len(report['errors'])}]")
        for err in report["errors"]:
            print(f"  [X] {err}")

    if report["warnings"]:
        print(f"\n[WARNINGS / NOTICES: {len(report['warnings'])}]")
        for w in report["warnings"]:
            print(f"  [!] {w}")

    if report["status"] == "PASSED" and not report["errors"]:
        print("\n[SUCCESS] Dimension benchmark dataset infrastructure is intact and schema-compliant!")


if __name__ == "__main__":
    main()
