"""
audit_expanded_dataset.py - Comprehensive Dataset Audit and Validation Suite
============================================================================
Project: Cargo Vision Logistics System
Purpose: Rigorously audits any YOLO-formatted dataset (e.g., deployment_dataset/
         or deployment_dataset_expanded/) verifying 24-class taxonomy integrity,
         per-split instance distributions, bounding box validity within [0,1],
         cross-split image deduplication, and resolution profiles.
"""

import os
import sys
import yaml
import json
import hashlib
import argparse
from typing import Dict, Any, List, Set, Tuple
from collections import Counter, defaultdict
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXPECTED_24_CLASSES = [
    "Bed", "Box", "Chair", "Couch", "Desk", "Door", "Drawer", "Laundry",
    "Person", "Shoe", "Sink", "Suitcase", "Table", "Tv", "book shelf",
    "cat", "dog", "fan", "mirror", "refrigerator", "stool", "stove",
    "toilet", "trashcan"
]


def audit_yolo_dataset(
    dataset_dir: str = "deployment_dataset",
    yaml_name: str = "data.yaml",
    output_json: str = "results/dataset_audit_report.json",
) -> Dict[str, Any]:
    """
    Executes a comprehensive, non-destructive audit of a YOLO dataset directory.
    """
    dataset_path = os.path.abspath(dataset_dir)
    yaml_path = os.path.join(dataset_path, yaml_name)

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Dataset YAML configuration not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    class_names = data_cfg.get("names", [])
    nc = len(class_names)

    splits = ["train", "valid", "test"]
    split_stats: Dict[str, Dict[str, int]] = {}
    class_instance_counts: Dict[str, Counter] = {s: Counter() for s in splits}
    class_image_counts: Dict[str, Counter] = {s: Counter() for s in splits}
    total_instances_per_class: Counter = Counter()

    empty_label_files: Dict[str, List[str]] = {s: [] for s in splits}
    invalid_boxes: Dict[str, List[str]] = {s: [] for s in splits}
    malformed_lines: Dict[str, List[str]] = {s: [] for s in splits}
    mismatched_pairs: Dict[str, Dict[str, List[str]]] = {s: {} for s in splits}

    image_hashes: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    resolution_counts: Counter = Counter()
    split_filenames: Dict[str, Set[str]] = {s: set() for s in splits}

    total_dataset_images = 0
    total_dataset_instances = 0

    for s in splits:
        img_dir = os.path.join(dataset_path, s, "images")
        lbl_dir = os.path.join(dataset_path, s, "labels")

        img_files = os.listdir(img_dir) if os.path.exists(img_dir) else []
        lbl_files = os.listdir(lbl_dir) if os.path.exists(lbl_dir) else []

        img_bases = {os.path.splitext(f)[0]: f for f in img_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))}
        lbl_bases = {os.path.splitext(f)[0]: f for f in lbl_files if f.endswith(".txt")}

        split_filenames[s] = set(img_files)
        total_dataset_images += len(img_bases)

        # Check pairing
        missing_labels = [f for b, f in img_bases.items() if b not in lbl_bases]
        orphan_labels = [f for b, f in lbl_bases.items() if b not in img_bases]
        if missing_labels or orphan_labels:
            mismatched_pairs[s] = {
                "images_without_labels": missing_labels,
                "labels_without_images": orphan_labels,
            }

        split_instances = 0

        # Audit annotations
        for b, lbl_fname in lbl_bases.items():
            lbl_p = os.path.join(lbl_dir, lbl_fname)
            classes_in_image: Set[str] = set()

            with open(lbl_p, "r", encoding="utf-8") as lf:
                lines = [l.strip() for l in lf.readlines() if l.strip()]

            if len(lines) == 0:
                empty_label_files[s].append(lbl_fname)

            for line_idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 5:
                    malformed_lines[s].append(f"{lbl_fname}:L{line_idx+1} (insufficient tokens: {len(parts)})")
                    continue

                try:
                    cls_id = int(parts[0])
                    if not (0 <= cls_id < nc):
                        invalid_boxes[s].append(f"{lbl_fname}:L{line_idx+1} (invalid class index {cls_id})")
                        continue

                    cname = class_names[cls_id]

                    # Standard YOLO box (5 tokens) or polygon segmentation (>5 tokens)
                    if len(parts) == 5:
                        xc, yc, w, h = map(float, parts[1:])
                        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                            invalid_boxes[s].append(f"{lbl_fname}:L{line_idx+1} (box out of bounds [0,1]: xc={xc}, yc={yc}, w={w}, h={h})")
                        if w <= 0.0 or h <= 0.0:
                            invalid_boxes[s].append(f"{lbl_fname}:L{line_idx+1} (non-positive dimension: w={w}, h={h})")
                    else:
                        coords = list(map(float, parts[1:]))
                        for coord in coords:
                            if not (0.0 <= coord <= 1.0):
                                invalid_boxes[s].append(f"{lbl_fname}:L{line_idx+1} (polygon vertex out of bounds [0,1]: {coord})")
                                break

                    class_instance_counts[s][cname] += 1
                    total_instances_per_class[cname] += 1
                    split_instances += 1
                    total_dataset_instances += 1
                    classes_in_image.add(cname)

                except Exception as ex:
                    malformed_lines[s].append(f"{lbl_fname}:L{line_idx+1} ({str(ex)})")

            for cname in classes_in_image:
                class_image_counts[s][cname] += 1

        # Audit image hashes & resolutions
        for b, img_fname in img_bases.items():
            img_p = os.path.join(img_dir, img_fname)
            try:
                with open(img_p, "rb") as imf:
                    h_val = hashlib.sha256(imf.read()).hexdigest()
                    image_hashes[h_val].append((s, img_fname))

                with Image.open(img_p) as im:
                    resolution_counts[f"{im.width}x{im.height}"] += 1
            except Exception as ex:
                malformed_lines[s].append(f"{img_fname} (image open error: {str(ex)})")

        split_stats[s] = {
            "images_count": len(img_bases),
            "label_files_count": len(lbl_bases),
            "annotated_instances_count": split_instances,
        }

    # Cross-split duplicate detection
    cross_split_duplicates = {}
    for h_val, occ_list in image_hashes.items():
        distinct_splits = set(occ[0] for occ in occ_list)
        if len(distinct_splits) > 1:
            cross_split_duplicates[h_val] = occ_list

    # Class summary table
    per_class_summary = []
    for cid, cname in enumerate(class_names):
        tot_i = total_instances_per_class[cname]
        tr_i = class_instance_counts["train"][cname]
        va_i = class_instance_counts["valid"][cname]
        te_i = class_instance_counts["test"][cname]
        tr_img = class_image_counts["train"][cname]
        va_img = class_image_counts["valid"][cname]
        te_img = class_image_counts["test"][cname]

        status = "HEALTHY"
        if tot_i < 10:
            status = "CRITICAL_SPARSE (<10 instances)"
        elif va_i == 0 or te_i == 0:
            status = "ZERO_VAL_OR_TEST"
        elif tr_i < 30:
            status = "LOW_TRAIN_SAMPLES"

        per_class_summary.append({
            "class_id": cid,
            "class_name": cname,
            "total_instances": tot_i,
            "train_instances": tr_i,
            "valid_instances": va_i,
            "test_instances": te_i,
            "train_images": tr_img,
            "valid_images": va_img,
            "test_images": te_img,
            "status": status,
        })

    audit_report = {
        "dataset_directory": dataset_path,
        "yaml_configuration": yaml_path,
        "classes_count": nc,
        "class_names": class_names,
        "total_images": total_dataset_images,
        "total_instances": total_dataset_instances,
        "split_statistics": split_stats,
        "per_class_summary": per_class_summary,
        "data_integrity": {
            "empty_label_files_count": sum(len(v) for v in empty_label_files.values()),
            "invalid_bounding_boxes_count": sum(len(v) for v in invalid_boxes.values()),
            "malformed_lines_count": sum(len(v) for v in malformed_lines.values()),
            "image_label_mismatches_count": sum(len(v) for v in mismatched_pairs.values()),
            "cross_split_duplicates_count": len(cross_split_duplicates),
        },
        "image_resolution_profile": dict(resolution_counts),
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)

    return audit_report


def print_audit_report(report: Dict[str, Any]):
    """Prints a formatted ASCII report of the dataset audit."""
    print("=" * 88)
    print("CARGO VISION - YOLO DATASET AUDIT & INTEGRITY REPORT")
    print("=" * 88)
    print(f"Dataset Path:       {report['dataset_directory']}")
    print(f"Classes Configured: {report['classes_count']}")
    print(f"Total Images:       {report['total_images']}")
    print(f"Total Annotations:  {report['total_instances']}")
    print("-" * 88)
    print("SPLIT SUMMARY:")
    for s, st in report["split_statistics"].items():
        print(f"  - Split '{s:<5}': {st['images_count']:<4} images | {st['label_files_count']:<4} label files | {st['annotated_instances_count']:<4} annotations")
    print("-" * 88)
    print(f"{'ID':<3} | {'CLASS NAME':<14} | {'TOTAL':<6} | {'TRAIN':<6} | {'VAL':<5} | {'TEST':<5} | {'TR_IMG':<6} | {'STATUS':<24}")
    print("-" * 88)
    for c in report["per_class_summary"]:
        print(f"{c['class_id']:<3} | {c['class_name']:<14} | {c['total_instances']:<6} | {c['train_instances']:<6} | {c['valid_instances']:<5} | {c['test_instances']:<5} | {c['train_images']:<6} | {c['status']:<24}")
    print("-" * 88)
    integ = report["data_integrity"]
    print("DATA INTEGRITY:")
    print(f"  - Empty Label Files:          {integ['empty_label_files_count']}")
    print(f"  - Invalid Bounding Boxes:     {integ['invalid_bounding_boxes_count']}")
    print(f"  - Malformed Lines:            {integ['malformed_lines_count']}")
    print(f"  - Image-Label Mismatches:     {integ['image_label_mismatches_count']}")
    print(f"  - Cross-Split Duplicate Leaks:{integ['cross_split_duplicates_count']}")
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser(description="Audit YOLO Dataset for Cargo Vision.")
    parser.add_argument("--dataset", type=str, default="deployment_dataset", help="Dataset directory.")
    parser.add_argument("--yaml", type=str, default="data.yaml", help="YAML file name.")
    parser.add_argument("--output", type=str, default="results/dataset_audit_report.json", help="Output JSON report path.")
    args = parser.parse_args()

    report = audit_yolo_dataset(dataset_dir=args.dataset, yaml_name=args.yaml, output_json=args.output)
    print_audit_report(report)


if __name__ == "__main__":
    main()
