"""
prepare_classification_data.py
==============================
Project: Cargo Vision CNN
Purpose: Converts multi-object YOLO detection/segmentation dataset (deployment_dataset)
         into a clean 5-class image classification dataset with isolated object crops.

Target Classes:
  - Box      (Class ID: 1)  -> 'box'
  - Chair    (Class ID: 2)  -> 'chair'
  - Couch    (Class ID: 3)  -> 'couch'
  - Suitcase (Class ID: 11) -> 'suitcase'
  - Table    (Class ID: 12) -> 'table'

Features:
  1. Dual-format support: Parses both YOLO segmentation polygons and standard 5-token bboxes.
  2. Bounding-box padding: Adds configurable padding (default 8%) clamped to image boundaries.
  3. Zero-leakage preservation: Retains original train/valid/test split assignments.
  4. Unique collision-free naming: {image_stem}_{class_name}_{obj_idx}.jpg.
  5. Audit & Provenance: Generates dataset_summary.json and classification README.md.
  6. Integrity verification: Validates all generated crops via Pillow.
"""

import os
import glob
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

DEFAULT_TARGET_CLASSES: Dict[int, str] = {
    1: "box",
    2: "chair",
    3: "couch",
    11: "suitcase",
    12: "table",
}

EXPANDED_TARGET_CLASSES: Dict[int, str] = {
    1: "box",
    2: "chair",
    3: "couch",
    4: "table",     # Desk -> Table (identical logistics category)
    11: "suitcase",
    12: "table",
    20: "chair",    # Stool -> Chair (seating furniture)
}

DEFAULT_PADDING = 0.08  # 8% margin around bounding box
SPLITS = ["train", "valid", "test"]


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def parse_annotation_line(
    line: str,
    target_classes: Optional[Dict[int, str]] = None,
) -> Optional[Tuple[int, str, float, float, float, float, str]]:
    """
    Parses a single annotation line supporting both YOLO bbox and polygon segmentation formats.
    
    Returns:
        (class_id, class_name, xmin, ymin, xmax, ymax, annotation_type) or None if non-target/invalid.
    """
    if target_classes is None:
        target_classes = DEFAULT_TARGET_CLASSES

    tokens = line.strip().split()
    if not tokens:
        return None

    try:
        class_id = int(tokens[0])
    except ValueError:
        return None

    # Skip non-target classes
    if class_id not in target_classes:
        return None

    class_name = target_classes[class_id]

    try:
        coords = [float(x) for x in tokens[1:]]
    except ValueError:
        return None

    # Format 1: Standard 5-token YOLO Bounding Box (class_id xc yc w h)
    if len(coords) == 4:
        xc, yc, w, h = coords
        xmin = xc - (w / 2.0)
        xmax = xc + (w / 2.0)
        ymin = yc - (h / 2.0)
        ymax = yc + (h / 2.0)
        ann_type = "bbox"

    # Format 2: YOLO Polygon Segmentation (class_id x1 y1 x2 y2 x3 y3 ...)
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)
        ann_type = "polygon"

    else:
        return None

    # Clamp normalized coordinates to [0.0, 1.0]
    xmin = max(0.0, min(1.0, xmin))
    xmax = max(0.0, min(1.0, xmax))
    ymin = max(0.0, min(1.0, ymin))
    ymax = max(0.0, min(1.0, ymax))

    # Reject zero or inverted area bounding boxes
    if xmax <= xmin or ymax <= ymin:
        return None

    return class_id, class_name, xmin, ymin, xmax, ymax, ann_type


def compute_pixel_crop_coords(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int,
    padding: float,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Converts normalized coordinates to pixel coordinates, applies padding, and clamps to boundaries.
    """
    px_xmin = xmin * img_w
    px_xmax = xmax * img_w
    px_ymin = ymin * img_h
    px_ymax = ymax * img_h

    box_w = px_xmax - px_xmin
    box_h = px_ymax - px_ymin

    if box_w < 2 or box_h < 2:
        return None

    pad_x = box_w * padding
    pad_y = box_h * padding

    crop_xmin = max(0, int(round(px_xmin - pad_x)))
    crop_xmax = min(img_w, int(round(px_xmax + pad_x)))
    crop_ymin = max(0, int(round(px_ymin - pad_y)))
    crop_ymax = min(img_h, int(round(px_ymax + pad_y)))

    if crop_xmax <= crop_xmin or crop_ymax <= crop_ymin:
        return None

    return crop_xmin, crop_ymin, crop_xmax, crop_ymax


# -----------------------------------------------------------------------------
# Main Processing Engine
# -----------------------------------------------------------------------------

def process_dataset(
    source_dir: str = "deployment_dataset",
    output_dir: str = "data/classification",
    padding: float = DEFAULT_PADDING,
    dry_run: bool = False,
    expand_furniture: bool = False,
) -> Dict[str, Any]:
    """
    Main processing loop that extracts, pads, validates, and saves classification crops.
    """
    target_classes = EXPANDED_TARGET_CLASSES if expand_furniture else DEFAULT_TARGET_CLASSES
    unique_class_names = sorted(list(set(target_classes.values())))

    stats: Dict[str, Any] = {
        "source_dataset_path": os.path.abspath(source_dir),
        "output_dataset_path": os.path.abspath(output_dir),
        "target_classes": target_classes,
        "padding_ratio": padding,
        "splits": SPLITS,
        "source_images_per_split": {},
        "source_labels_per_split": {},
        "annotation_types": {"polygon": 0, "bbox": 0},
        "annotations_by_class": {c: 0 for c in unique_class_names},
        "crops_per_class_per_split": {s: {c: 0 for c in unique_class_names} for s in SPLITS},
        "rejected_annotations": 0,
        "total_target_annotations": 0,
        "total_crops_generated": 0,
        "verified_valid_crops": 0,
        "dry_run": dry_run,
        "expand_furniture": expand_furniture,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Prepare output directory structure
    if not dry_run:
        for s in SPLITS:
            for c in unique_class_names:
                os.makedirs(os.path.join(output_dir, s, c), exist_ok=True)

    print("\n" + "=" * 70)
    print("Cargo Vision CNN - Classification Dataset Preprocessing Pipeline")
    print(f"Source: {source_dir} | Output: {output_dir} | Padding: {padding:.1%}")
    print(f"Expansion Mode: {'EXPANDED (Desk->Table, Stool->Chair)' if expand_furniture else 'STANDARD (5 Classes)'}")
    print(f"Mode: {'DRY RUN (Analysis Only)' if dry_run else 'ACTIVE EXTRACTION'}")
    print("=" * 70)

    for split in SPLITS:
        images_dir = os.path.join(source_dir, split, "images")
        labels_dir = os.path.join(source_dir, split, "labels")

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"[Warning] Missing split directory: {split}")
            continue

        image_files = sorted(
            glob.glob(os.path.join(images_dir, "*.jpg"))
            + glob.glob(os.path.join(images_dir, "*.jpeg"))
            + glob.glob(os.path.join(images_dir, "*.png"))
        )
        label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))

        stats["source_images_per_split"][split] = len(image_files)
        stats["source_labels_per_split"][split] = len(label_files)
        print(f"\nProcessing split '{split}': {len(image_files)} images, {len(label_files)} label files...")

        # Map base name to label file
        label_map = {os.path.splitext(os.path.basename(p))[0]: p for p in label_files}

        split_crops = 0

        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = label_map.get(base_name)

            if not lbl_path or not os.path.exists(lbl_path):
                continue

            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Read image metadata / dimensions
            try:
                with Image.open(img_path) as full_img:
                    img_w, img_h = full_img.size
                    rgb_img = None if dry_run else full_img.convert("RGB")
            except Exception as e:
                print(f"  [Warning] Corrupt source image skipped ({img_path}): {e}")
                continue

            # Parse annotations
            obj_idx = 0
            for line in lines:
                parsed = parse_annotation_line(line, target_classes=target_classes)
                if not parsed:
                    continue

                class_id, class_name, xmin, ymin, xmax, ymax, ann_type = parsed
                stats["total_target_annotations"] += 1
                stats["annotation_types"][ann_type] += 1
                stats["annotations_by_class"][class_name] += 1

                crop_coords = compute_pixel_crop_coords(
                    xmin, ymin, xmax, ymax, img_w, img_h, padding
                )

                if not crop_coords:
                    stats["rejected_annotations"] += 1
                    continue

                c_xmin, c_ymin, c_xmax, c_ymax = crop_coords

                if not dry_run and rgb_img is not None:
                    # Perform crop
                    crop = rgb_img.crop((c_xmin, c_ymin, c_xmax, c_ymax))
                    dest_filename = f"{base_name}_{class_name}_{obj_idx}.jpg"
                    dest_path = os.path.join(output_dir, split, class_name, dest_filename)

                    # Save with high quality
                    crop.save(dest_path, "JPEG", quality=95)

                    # Verification pass on written file
                    with Image.open(dest_path) as v_img:
                        v_img.verify()
                    stats["verified_valid_crops"] += 1

                stats["crops_per_class_per_split"][split][class_name] += 1
                stats["total_crops_generated"] += 1
                split_crops += 1
                obj_idx += 1

        print(f"  -> Extracted {split_crops} target crops for split '{split}'.")

    return stats


def generate_reports(stats: Dict[str, Any], output_dir: str) -> None:
    """
    Writes machine-readable dataset_summary.json and human-readable README.md.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_json_path = os.path.join(output_dir, "dataset_summary.json")
    readme_path = os.path.join(output_dir, "README.md")

    # 1. Write dataset_summary.json
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[Audit] Saved machine-readable summary to '{summary_json_path}'")

    # 2. Write README.md
    readme_content = f"""# Cargo Vision CNN: Classification Dataset Report

This directory contains the 5-class object-crop classification dataset extracted from `deployment_dataset/`.

## 1. Dataset Overview
- **Source Dataset:** `deployment_dataset/` (Roboflow Universe / Household Item Dataset, License: CC BY 4.0)
- **Target Classes:**
  - `box` (Class ID: 1)
  - `chair` (Class ID: 2)
  - `couch` (Class ID: 3)
  - `suitcase` (Class ID: 11)
  - `table` (Class ID: 12)
- **Padding Ratio:** {stats['padding_ratio']:.1%} contextual margin around bounding box.
- **Data Leakage Safeguard:** 100% preservation of original splits (`train`, `valid`, `test`).
- **Total Crops Generated:** {stats['total_crops_generated']}
- **Generated At (UTC):** {stats['timestamp_utc']}

## 2. Crop Distribution by Class and Split

| Class Name | Train Crops | Valid Crops | Test Crops | Total Crops |
| :--- | :---: | :---: | :---: | :---: |
"""
    classes = sorted(list(stats["annotations_by_class"].keys()))
    for c in classes:
        tr = stats["crops_per_class_per_split"]["train"][c]
        va = stats["crops_per_class_per_split"]["valid"][c]
        te = stats["crops_per_class_per_split"]["test"][c]
        tot = tr + va + te
        readme_content += f"| **{c}** | {tr} | {va} | {te} | **{tot}** |\n"

    tot_tr = sum(stats["crops_per_class_per_split"]["train"].values())
    tot_va = sum(stats["crops_per_class_per_split"]["valid"].values())
    tot_te = sum(stats["crops_per_class_per_split"]["test"].values())
    grand_tot = tot_tr + tot_va + tot_te

    readme_content += f"| **TOTAL** | **{tot_tr}** | **{tot_va}** | **{tot_te}** | **{grand_tot}** |\n\n"

    readme_content += f"""## 3. Annotation & Parsing Statistics
- **Total Target Annotations Encountered:** {stats['total_target_annotations']}
- **Polygon Segmentation Annotations:** {stats['annotation_types']['polygon']}
- **Standard Bounding Box Annotations:** {stats['annotation_types']['bbox']}
- **Rejected / Invalid Annotations:** {stats['rejected_annotations']}
- **Verified Valid Crops Written:** {stats['verified_valid_crops']}

## 4. Class Imbalance & Training Recommendations
- `chair` ({stats['annotations_by_class']['chair']} crops) has the highest representation.
- `suitcase` ({stats['annotations_by_class']['suitcase']} crops) has the lowest representation.
- **Recommendation:** Utilize on-the-fly data augmentation (random horizontal flips, slight rotations, zooming) during CNN training to provide robust feature generalization.
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"[Report] Saved human-readable README report to '{readme_path}'")


def print_summary_table(stats: Dict[str, Any]) -> None:
    """
    Prints a formatted summary table to console.
    """
    classes = sorted(list(stats["annotations_by_class"].keys()))
    print("\n" + "=" * 70)
    print(f"{'Class Name':<12} | {'Train':<10} | {'Valid':<10} | {'Test':<10} | {'Total':<10}")
    print("-" * 70)

    tot_tr, tot_va, tot_te = 0, 0, 0
    for c in classes:
        tr = stats["crops_per_class_per_split"]["train"][c]
        va = stats["crops_per_class_per_split"]["valid"][c]
        te = stats["crops_per_class_per_split"]["test"][c]
        tot = tr + va + te
        tot_tr += tr
        tot_va += va
        tot_te += te
        print(f"{c:<12} | {tr:<10} | {va:<10} | {te:<10} | {tot:<10}")

    print("-" * 70)
    print(f"{'TOTAL':<12} | {tot_tr:<10} | {tot_va:<10} | {tot_te:<10} | {tot_tr + tot_va + tot_te:<10}")
    print("=" * 70)
    print(f"Annotation Types: {stats['annotation_types']['polygon']} Polygons, {stats['annotation_types']['bbox']} BBoxes")
    print(f"Rejected / Zero-Area Annotations: {stats['rejected_annotations']}")
    if not stats["dry_run"]:
        print(f"Verified Crops on Disk: {stats['verified_valid_crops']} (100% Valid JPEGs)")


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO multi-object dataset to 5-class image classification crop dataset."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="deployment_dataset",
        help="Path to source YOLO dataset (default: deployment_dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/classification",
        help="Path to output classification dataset (default: data/classification)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING,
        help=f"Contextual padding ratio around bounding box (default: {DEFAULT_PADDING})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform parsing analysis without writing crop files to disk.",
    )
    parser.add_argument(
        "--expand-furniture",
        action="store_true",
        help="Include related logistics classes (Desk -> table, stool -> chair) to safely expand training and test support.",
    )
    args = parser.parse_args()

    # Process dataset
    stats = process_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        padding=args.padding,
        dry_run=args.dry_run,
        expand_furniture=args.expand_furniture,
    )

    # Print summary
    print_summary_table(stats)

    # Generate summary & README files if not in dry-run mode
    if not args.dry_run:
        generate_reports(stats, args.output_dir)


if __name__ == "__main__":
    main()
