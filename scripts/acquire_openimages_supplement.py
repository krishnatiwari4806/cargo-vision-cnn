"""
acquire_openimages_supplement.py - Official Google Open Images V7 Supplement Ingestion Engine
=============================================================================================
Project: Cargo Vision Logistics System
Purpose: Ingests additional real images and official ground-truth bounding box annotations
         from Google Open Images V7 for the 24-class Cargo Vision taxonomy.
         Stages all new acquisitions into an isolated directory (data/openimages_supplemental/)
         without modifying deployment_dataset/ or deployment_dataset_expanded/.
"""

import os
import sys
import csv
import json
import yaml
import shutil
import hashlib
import random
import requests
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Canonical 24-Class Taxonomy
CANONICAL_24_CLASSES = [
    "Bed", "Box", "Chair", "Couch", "Desk", "Door", "Drawer", "Laundry",
    "Person", "Shoe", "Sink", "Suitcase", "Table", "Tv", "book shelf",
    "cat", "dog", "fan", "mirror", "refrigerator", "stool", "stove",
    "toilet", "trashcan"
]

CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(CANONICAL_24_CLASSES)}

# Complete OpenImages V7 MID mappings for all 24 classes
OPENIMAGES_MID_TO_CLASS = {
    "/m/03ssj5": ("Bed", "Bed"),
    "/m/025dyy": ("Box", "Box"),
    "/m/01mzpv": ("Chair", "Chair"),
    "/m/02crq1": ("Couch", "Couch"),
    "/m/03m3pdh": ("Couch", "Sofa bed"),
    "/m/026qbn5": ("Couch", "Studio couch"),
    "/m/01y9k5": ("Desk", "Desk"),
    "/m/02dgv": ("Door", "Door"),
    "/m/0fqfqc": ("Drawer", "Drawer"),
    "/m/05kyg_": ("Drawer", "Chest of drawers"),
    "/m/09j2d": ("Laundry", "Clothing"),
    "/m/01g317": ("Person", "Person"),
    "/m/09j5n": ("Shoe", "Footwear"),
    "/m/06k2mb": ("Shoe", "High heels"),
    "/m/01b638": ("Shoe", "Boot"),
    "/m/0130jx": ("Sink", "Sink"),
    "/m/01s55n": ("Suitcase", "Suitcase"),
    "/m/0hf58v5": ("Suitcase", "Luggage and bags"),
    "/m/04bcr3": ("Table", "Table"),
    "/m/078n6m": ("Table", "Coffee table"),
    "/m/0h8n5zk": ("Table", "Kitchen & dining room table"),
    "/m/07c52": ("Tv", "Television"),
    "/m/03__z0": ("book shelf", "Bookcase"),
    "/m/01yrx": ("cat", "Cat"),
    "/m/0bt9lr": ("dog", "Dog"),
    "/m/03ldnb": ("fan", "Ceiling fan"),
    "/m/02x984l": ("fan", "Mechanical fan"),
    "/m/054_l": ("mirror", "Mirror"),
    "/m/040b_t": ("refrigerator", "Refrigerator"),
    "/m/0fqt361": ("stool", "Stool"),
    "/m/02wv84t": ("stove", "Gas stove"),
    "/m/029bxz": ("stove", "Oven"),
    "/m/09g1w": ("toilet", "Toilet"),
    "/m/0bjyj5": ("trashcan", "Waste container"),
}


def validate_yolo_bbox(
    class_id: int, xc: float, yc: float, w: float, h: float, nc: int = 24
) -> Tuple[bool, str]:
    """Validates YOLO bounding box constraints in [0.0, 1.0]."""
    if not (0 <= class_id < nc):
        return False, f"Class ID {class_id} out of bounds [0, {nc-1}]"
    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
        return False, f"Coordinates out of [0, 1]: xc={xc}, yc={yc}, w={w}, h={h}"
    if w <= 1e-6 or h <= 1e-6:
        return False, f"Non-positive dimensions: w={w}, h={h}"
    return True, "Valid"


def compute_image_sha256(image_path: str) -> str:
    """Computes SHA-256 hash of an image file."""
    hasher = hashlib.sha256()
    with open(image_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class OpenImagesSupplementAcquirer:
    """
    Acquires, validates, deduplicates, and stages real Open Images V7 samples.
    """

    def __init__(
        self,
        supplement_dir: str = "data/openimages_supplemental",
        cache_dir: str = "data/public_cache",
        seed: int = 42,
    ):
        self.supplement_dir = os.path.abspath(supplement_dir)
        self.cache_dir = os.path.abspath(cache_dir)
        self.seed = seed
        self.existing_hashes: Set[str] = set()
        self.existing_image_ids: Set[str] = set()
        self.acquired_samples: List[Dict[str, Any]] = []

    def scan_existing_datasets_to_prevent_duplicates(self):
        """Scans deployment_dataset and deployment_dataset_expanded to guarantee zero duplicate ingestion."""
        # 1. Check deployment_dataset
        orig_dir = os.path.join(PROJECT_ROOT, "deployment_dataset")
        if os.path.exists(orig_dir):
            for s in ["train", "valid", "test"]:
                im_d = os.path.join(orig_dir, s, "images")
                if os.path.exists(im_d):
                    for im_f in os.listdir(im_d):
                        if im_f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                            p = os.path.join(im_d, im_f)
                            self.existing_hashes.add(compute_image_sha256(p))

        # 2. Check deployment_dataset_expanded
        exp_dir = os.path.join(PROJECT_ROOT, "deployment_dataset_expanded")
        if os.path.exists(exp_dir):
            for s in ["train", "valid", "test"]:
                im_d = os.path.join(exp_dir, s, "images")
                if os.path.exists(im_d):
                    for im_f in os.listdir(im_d):
                        if im_f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                            p = os.path.join(im_d, im_f)
                            self.existing_hashes.add(compute_image_sha256(p))

        # 3. Check provenance records
        prov_path = os.path.join(exp_dir, "metadata", "provenance.json")
        if os.path.exists(prov_path):
            with open(prov_path, "r", encoding="utf-8") as pf:
                pdata = json.load(pf)
                for rec in pdata.get("provenance_records", []):
                    self.existing_image_ids.add(rec.get("source_image_id"))

        print(f"[OK] Indexed {len(self.existing_hashes)} existing image hashes and {len(self.existing_image_ids)} provenance IDs to prevent duplicates.")

    def acquire_samples(
        self,
        max_samples_per_class: int = 35,
        target_total_samples: int = 350,
    ) -> Dict[str, Any]:
        """
        Extracts new Open Images V7 annotations, downloads images, verifies them, and writes to supplement_dir.
        """
        random.seed(self.seed)
        self.scan_existing_datasets_to_prevent_duplicates()

        oid_csv_path = os.path.join(self.cache_dir, "openimages_validation_bbox.csv")
        if not os.path.exists(oid_csv_path):
            raise FileNotFoundError(f"OpenImages validation CSV not found at: {oid_csv_path}")

        img_to_boxes = defaultdict(list)
        class_to_images = defaultdict(set)

        with open(oid_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = row["LabelName"]
                if mid in OPENIMAGES_MID_TO_CLASS:
                    our_cls, orig_label = OPENIMAGES_MID_TO_CLASS[mid]
                    img_id = row["ImageID"]
                    if img_id in self.existing_image_ids:
                        continue
                    try:
                        xmin = float(row["XMin"])
                        xmax = float(row["XMax"])
                        ymin = float(row["YMin"])
                        ymax = float(row["YMax"])
                        w = max(0.0, xmax - xmin)
                        h = max(0.0, ymax - ymin)
                        xc = xmin + w / 2.0
                        yc = ymin + h / 2.0
                        target_id = CLASS_NAME_TO_ID[our_cls]
                        is_val, _ = validate_yolo_bbox(target_id, xc, yc, w, h)
                        if is_val:
                            img_to_boxes[img_id].append({
                                "class_id": target_id,
                                "class_name": our_cls,
                                "orig_label": orig_label,
                                "mid": mid,
                                "bbox": (xc, yc, w, h),
                            })
                            class_to_images[our_cls].add(img_id)
                    except ValueError:
                        continue

        # Priority selection across classes
        selected_image_ids: Set[str] = set()

        # Prioritize rare cargo and underrepresented items
        priority_classes = [
            "trashcan", "stool", "fan", "mirror", "toilet", "refrigerator",
            "Sink", "Suitcase", "stove", "book shelf", "Box", "Desk",
            "Drawer", "Door", "Couch", "Table", "Bed", "Tv", "Chair",
            "cat", "dog", "Shoe", "Laundry", "Person"
        ]

        for cname in priority_classes:
            if cname in class_to_images:
                c_img_ids = sorted(list(class_to_images[cname]))
                random.shuffle(c_img_ids)
                cap = max_samples_per_class
                if cname in ["Person", "Laundry", "Shoe"]:
                    cap = min(cap, 15)  # Cap dominant classes

                for img_id in c_img_ids:
                    if img_id not in selected_image_ids:
                        selected_image_ids.add(img_id)
                        if len(selected_image_ids) >= target_total_samples:
                            break
            if len(selected_image_ids) >= target_total_samples:
                break

        print(f"Downloading and validating {len(selected_image_ids)} supplemental Open Images V7 samples...")

        # Setup staging directory
        shutil.rmtree(self.supplement_dir, ignore_errors=True)
        out_img_dir = os.path.join(self.supplement_dir, "images")
        out_lbl_dir = os.path.join(self.supplement_dir, "labels")
        out_meta_dir = os.path.join(self.supplement_dir, "metadata")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)
        os.makedirs(out_meta_dir, exist_ok=True)

        session = requests.Session()
        acquired_records = []
        new_annotations_per_class = Counter()
        new_images_per_class = Counter()
        successful_images = 0

        for img_id in selected_image_ids:
            oid_url = f"https://open-images-dataset.s3.amazonaws.com/validation/{img_id}.jpg"
            temp_img_p = os.path.join(out_img_dir, f"oid_supp_{img_id}.jpg")

            try:
                resp = session.get(oid_url, timeout=15)
                if resp.status_code == 200:
                    with open(temp_img_p, "wb") as f:
                        f.write(resp.content)
                else:
                    continue
            except Exception:
                continue

            # Verify image file
            try:
                with Image.open(temp_img_p) as im:
                    im.verify()
            except Exception:
                if os.path.exists(temp_img_p):
                    os.remove(temp_img_p)
                continue

            # Check SHA-256 hash deduplication
            img_hash = compute_image_sha256(temp_img_p)
            if img_hash in self.existing_hashes:
                os.remove(temp_img_p)
                continue

            self.existing_hashes.add(img_hash)

            # Write YOLO label file
            boxes_info = img_to_boxes[img_id]
            lbl_p = os.path.join(out_lbl_dir, f"oid_supp_{img_id}.txt")
            classes_in_this_image = set()

            with open(lbl_p, "w", encoding="utf-8") as lf:
                for b_item in boxes_info:
                    cid = b_item["class_id"]
                    cname = b_item["class_name"]
                    xc, yc, w, h = b_item["bbox"]
                    lf.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    new_annotations_per_class[cname] += 1
                    classes_in_this_image.add(cname)

                    acquired_records.append({
                        "imported_filename": f"oid_supp_{img_id}.jpg",
                        "source_dataset": "Google_OpenImages_V7_val",
                        "source_image_id": img_id,
                        "openimages_mid": b_item["mid"],
                        "original_label": b_item["orig_label"],
                        "mapped_project_class": cname,
                        "class_id": cid,
                        "license": "Creative_Commons_Attribution_4.0_CC_BY",
                        "bbox_yolo": [round(xc, 6), round(yc, 6), round(w, 6), round(h, 6)],
                        "sha256_hash": img_hash,
                    })

            for cname in classes_in_this_image:
                new_images_per_class[cname] += 1

            successful_images += 1

        summary = {
            "total_new_images_acquired": successful_images,
            "total_new_annotations_acquired": sum(new_annotations_per_class.values()),
            "new_annotations_per_class": dict(new_annotations_per_class),
            "new_images_per_class": dict(new_images_per_class),
            "source_dataset": "Google_OpenImages_V7 (Official Google Cloud & S3 mirrors)",
            "license": "Creative Commons Attribution 4.0 (CC BY 4.0) / CC BY 2.0",
            "staging_directory": self.supplement_dir,
            "provenance_records": acquired_records,
        }

        # Write metadata
        with open(os.path.join(out_meta_dir, "supplement_provenance.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(PROJECT_ROOT, "results", "openimages_supplement_report.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[SUCCESS] Staged {successful_images} NEW Open Images V7 images with {sum(new_annotations_per_class.values())} annotations.")
        return summary


def main():
    acquirer = OpenImagesSupplementAcquirer(
        supplement_dir="data/openimages_supplemental",
        cache_dir="data/public_cache",
    )
    summary = acquirer.acquire_samples(max_samples_per_class=35, target_total_samples=300)
    print("\nPER-CLASS NEW ANNOTATION COUNTS:")
    for cname, count in sorted(summary["new_annotations_per_class"].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cname:<16}: {count} new annotations across {summary['new_images_per_class'].get(cname, 0)} images")


if __name__ == "__main__":
    main()
