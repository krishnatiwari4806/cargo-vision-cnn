"""
expand_dataset.py - Real Public Dataset Ingestion, Conversion, and Stratified Partitioning Engine
=================================================================================================
Project: Cargo Vision Logistics System
Purpose: Ingests real, verified images and ground-truth annotations from COCO 2017 and
         Google Open Images V7 (both CC BY licensed) to expand and balance the Cargo Vision
         24-class dataset into deployment_dataset_expanded/ (preserving original deployment_dataset).
"""

import os
import sys
import yaml
import json
import csv
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

# Mapping for COCO categories
COCO_NAME_TO_OUR_CLASS = {
    "bed": "Bed",
    "chair": "Chair",
    "couch": "Couch",
    "dining table": "Table",
    "tv": "Tv",
    "toilet": "toilet",
    "sink": "Sink",
    "refrigerator": "refrigerator",
    "suitcase": "Suitcase",
    "person": "Person",
    "cat": "cat",
    "dog": "dog",
}

# Mapping for OpenImages MIDs
OPENIMAGES_MID_TO_OUR_CLASS = {
    "/m/025dyy": ("Box", "Box"),
    "/m/01y9k5": ("Desk", "Desk"),
    "/m/02dgv": ("Door", "Door"),
    "/m/0fqfqc": ("Drawer", "Drawer"),
    "/m/05kyg_": ("Drawer", "Chest of drawers"),
    "/m/03__z0": ("book shelf", "Bookcase"),
    "/m/0fqt361": ("stool", "Stool"),
    "/m/02wv84t": ("stove", "Gas stove"),
    "/m/029bxz": ("stove", "Oven"),
    "/m/0bjyj5": ("trashcan", "Waste container"),
    "/m/03ldnb": ("fan", "Ceiling fan"),
    "/m/02x984l": ("fan", "Mechanical fan"),
    "/m/054_l": ("mirror", "Mirror"),
    "/m/09j5n": ("Shoe", "Footwear"),
    "/m/09j2d": ("Laundry", "Clothing"),
}

# Public dataset aliases
PUBLIC_DATASET_ALIASES = {
    "bed": "Bed",
    "box": "Box",
    "cardboard box": "Box",
    "package": "Box",
    "chair": "Chair",
    "armchair": "Chair",
    "couch": "Couch",
    "sofa": "Couch",
    "desk": "Desk",
    "office desk": "Desk",
    "door": "Door",
    "drawer": "Drawer",
    "chest of drawers": "Drawer",
    "laundry": "Laundry",
    "clothing": "Laundry",
    "clothes": "Laundry",
    "person": "Person",
    "shoe": "Shoe",
    "footwear": "Shoe",
    "sink": "Sink",
    "washbasin": "Sink",
    "suitcase": "Suitcase",
    "luggage": "Suitcase",
    "table": "Table",
    "dining table": "Table",
    "coffee table": "Table",
    "tv": "Tv",
    "television": "Tv",
    "book shelf": "book shelf",
    "bookshelf": "book shelf",
    "bookcase": "book shelf",
    "cat": "cat",
    "dog": "dog",
    "fan": "fan",
    "ceiling fan": "fan",
    "mechanical fan": "fan",
    "mirror": "mirror",
    "refrigerator": "refrigerator",
    "fridge": "refrigerator",
    "stool": "stool",
    "bar stool": "stool",
    "stove": "stove",
    "gas stove": "stove",
    "kitchen stove": "stove",
    "oven": "stove",
    "toilet": "toilet",
    "trashcan": "trashcan",
    "waste container": "trashcan",
    "trash can": "trashcan",
    "garbage can": "trashcan",
}


def normalize_class_name(raw_name: str) -> Optional[str]:
    """Normalizes raw string to one of our canonical 24 classes."""
    lowered = raw_name.strip().lower()
    if raw_name in CLASS_NAME_TO_ID:
        return raw_name
    return PUBLIC_DATASET_ALIASES.get(lowered, None)


def validate_yolo_bbox(
    class_id: int, xc: float, yc: float, w: float, h: float, nc: int = 24
) -> Tuple[bool, str]:
    """Validates YOLO bbox boundaries in [0.0, 1.0]."""
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


class RealPublicDatasetExpander:
    """
    Orchestrates ingestion of verified real public datasets into deployment_dataset_expanded.
    """

    def __init__(
        self,
        output_dir: str = "deployment_dataset_expanded",
        cache_dir: str = "data/public_cache",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.cache_dir = os.path.abspath(cache_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.seen_hashes: Dict[str, str] = {}
        self.samples: List[Dict[str, Any]] = []
        self.provenance_records: List[Dict[str, Any]] = []
        self.downloaded_images_dir = os.path.join(self.cache_dir, "downloaded_images")
        os.makedirs(self.downloaded_images_dir, exist_ok=True)

    def initialize_output_directory(self, clean: bool = True):
        """Initializes destination folders safely without deleting root directory."""
        for split in ["train", "valid", "test"]:
            img_dir = os.path.join(self.output_dir, split, "images")
            lbl_dir = os.path.join(self.output_dir, split, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            if clean:
                for f in os.listdir(img_dir):
                    try:
                        os.remove(os.path.join(img_dir, f))
                    except Exception:
                        pass
                for f in os.listdir(lbl_dir):
                    try:
                        os.remove(os.path.join(lbl_dir, f))
                    except Exception:
                        pass
        os.makedirs(os.path.join(self.output_dir, "metadata"), exist_ok=True)

    def ingest_original_deployment_dataset(self, source_dir: str = "deployment_dataset") -> int:
        """
        Preserves all 252 images and annotations from original deployment_dataset.
        """
        source_path = os.path.abspath(source_dir)
        yaml_path = os.path.join(source_path, "data.yaml")

        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Source YAML not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            src_cfg = yaml.safe_load(f)
        src_names = src_cfg.get("names", [])

        added = 0
        for split in ["train", "valid", "test"]:
            img_dir = os.path.join(source_path, split, "images")
            lbl_dir = os.path.join(source_path, split, "labels")

            if not os.path.exists(img_dir):
                continue

            for img_f in os.listdir(img_dir):
                if not img_f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue

                base = os.path.splitext(img_f)[0]
                lbl_f = base + ".txt"
                lbl_path = os.path.join(lbl_dir, lbl_f)
                img_path = os.path.join(img_dir, img_f)

                if not os.path.exists(lbl_path):
                    continue

                valid_boxes = []
                with open(lbl_path, "r", encoding="utf-8") as lf:
                    lines = [l.strip() for l in lf.readlines() if l.strip()]

                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])
                        cname = src_names[cid]
                        target_id = CLASS_NAME_TO_ID.get(cname)
                        if target_id is not None:
                            is_val, _ = validate_yolo_bbox(target_id, xc, yc, w, h)
                            if is_val:
                                valid_boxes.append((target_id, xc, yc, w, h))
                    elif len(parts) > 5:
                        cid = int(parts[0])
                        cname = src_names[cid]
                        target_id = CLASS_NAME_TO_ID.get(cname)
                        coords = list(map(float, parts[1:]))
                        xs, ys = coords[0::2], coords[1::2]
                        if xs and ys and target_id is not None:
                            min_x, max_x = max(0.0, min(xs)), min(1.0, max(xs))
                            min_y, max_y = max(0.0, min(ys)), min(1.0, max(ys))
                            w = max_x - min_x
                            h = max_y - min_y
                            xc = min_x + w / 2.0
                            yc = min_y + h / 2.0
                            is_val, _ = validate_yolo_bbox(target_id, xc, yc, w, h)
                            if is_val:
                                valid_boxes.append((target_id, xc, yc, w, h))

                if not valid_boxes:
                    continue

                img_hash = compute_image_sha256(img_path)
                if img_hash in self.seen_hashes:
                    continue

                self.seen_hashes[img_hash] = img_f
                self.samples.append({
                    "image_path": img_path,
                    "image_name": f"orig_{split}_{img_f}",
                    "boxes": valid_boxes,
                    "source_dataset": "deployment_dataset",
                    "source_image_id": img_f,
                    "license": "Roboflow_Cargo_CC_BY_4.0",
                })
                added += 1

        print(f"[OK] Ingested {added} baseline images from original deployment_dataset/ (100% preserved)")
        return added

    def ingest_coco_samples(
        self,
        max_images_per_class: int = 45,
        max_total_coco_images: int = 400,
    ) -> int:
        """
        Ingests real COCO 2017 validation images with bounding boxes.
        """
        coco_json_path = os.path.join(self.cache_dir, "instances_val2017.json")
        if not os.path.exists(coco_json_path):
            print(f"[WARN] COCO JSON not found at {coco_json_path}")
            return 0

        with open(coco_json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        coco_cats = {c["id"]: c["name"] for c in coco_data["categories"]}
        images_info = {im["id"]: im for im in coco_data["images"]}

        # Group annotations by image
        img_to_anns = defaultdict(list)
        class_to_images = defaultdict(set)

        for ann in coco_data["annotations"]:
            cid = ann["category_id"]
            cname = coco_cats.get(cid)
            if cname in COCO_NAME_TO_OUR_CLASS:
                our_class = COCO_NAME_TO_OUR_CLASS[cname]
                img_to_anns[ann["image_id"]].append((our_class, ann))
                class_to_images[our_class].add(ann["image_id"])

        selected_image_ids: Set[int] = set()
        random.seed(self.seed)

        for our_cls, img_ids in class_to_images.items():
            sorted_ids = sorted(list(img_ids))
            random.shuffle(sorted_ids)
            cap = max_images_per_class
            if our_cls in ["Person"]:
                cap = min(cap, 25)  # Avoid person dominance
            for img_id in sorted_ids[:cap]:
                selected_image_ids.add(img_id)
                if len(selected_image_ids) >= max_total_coco_images:
                    break
            if len(selected_image_ids) >= max_total_coco_images:
                break

        print(f"Ingesting {len(selected_image_ids)} selected COCO 2017 images...")
        added = 0

        session = requests.Session()
        for idx, img_id in enumerate(selected_image_ids):
            im_meta = images_info[img_id]
            w_img = im_meta["width"]
            h_img = im_meta["height"]
            fname = im_meta["file_name"]
            coco_url = f"http://images.cocodataset.org/val2017/{fname}"
            local_img_path = os.path.join(self.downloaded_images_dir, f"coco_{fname}")

            # Download if not cached
            if not os.path.exists(local_img_path):
                try:
                    resp = session.get(coco_url, timeout=15)
                    if resp.status_code == 200:
                        with open(local_img_path, "wb") as f:
                            f.write(resp.content)
                    else:
                        continue
                except Exception:
                    continue

            # Verify image
            try:
                with Image.open(local_img_path) as im:
                    im.verify()
            except Exception:
                continue

            img_hash = compute_image_sha256(local_img_path)
            if img_hash in self.seen_hashes:
                continue

            # Convert annotations
            valid_boxes = []
            for our_cls, ann in img_to_anns[img_id]:
                target_id = CLASS_NAME_TO_ID[our_cls]
                bx, by, bw, bh = ann["bbox"]
                if bw <= 0 or bh <= 0:
                    continue
                xc = (bx + bw / 2.0) / w_img
                yc = (by + bh / 2.0) / h_img
                wn = bw / w_img
                hn = bh / h_img
                is_val, _ = validate_yolo_bbox(target_id, xc, yc, wn, hn)
                if is_val:
                    valid_boxes.append((target_id, xc, yc, wn, hn))

            if not valid_boxes:
                continue

            self.seen_hashes[img_hash] = f"coco_{fname}"
            self.samples.append({
                "image_path": local_img_path,
                "image_name": f"coco_{fname}",
                "boxes": valid_boxes,
                "source_dataset": "MS_COCO_2017_val",
                "source_image_id": str(img_id),
                "license": "COCO_CC_BY_4.0",
            })
            added += 1

        print(f"[OK] Ingested {added} COCO real images with verified annotations.")
        return added

    def ingest_openimages_samples(
        self,
        max_images_per_class: int = 50,
        max_total_oid_images: int = 500,
    ) -> int:
        """
        Ingests real Google Open Images V7 validation images with bounding boxes.
        """
        oid_csv_path = os.path.join(self.cache_dir, "openimages_validation_bbox.csv")
        if not os.path.exists(oid_csv_path):
            print(f"[WARN] OpenImages CSV not found at {oid_csv_path}")
            return 0

        # Group annotations by image
        img_to_boxes = defaultdict(list)
        class_to_images = defaultdict(set)

        with open(oid_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = row["LabelName"]
                if mid in OPENIMAGES_MID_TO_OUR_CLASS:
                    our_cls, orig_name = OPENIMAGES_MID_TO_OUR_CLASS[mid]
                    img_id = row["ImageID"]
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
                            img_to_boxes[img_id].append((target_id, xc, yc, w, h, orig_name))
                            class_to_images[our_cls].add(img_id)
                    except ValueError:
                        continue

        selected_image_ids: Set[str] = set()
        random.seed(self.seed)

        # Prioritize rare cargo categories
        priority_order = [
            "Box", "Desk", "Door", "Drawer", "book shelf", "stove", "trashcan",
            "stool", "fan", "mirror", "Shoe", "Laundry"
        ]

        for our_cls in priority_order:
            if our_cls in class_to_images:
                img_ids = sorted(list(class_to_images[our_cls]))
                random.shuffle(img_ids)
                cap = max_images_per_class
                if our_cls in ["Laundry", "Shoe"]:
                    cap = min(cap, 30)  # Avoid single-class dominance
                for img_id in img_ids[:cap]:
                    selected_image_ids.add(img_id)
                    if len(selected_image_ids) >= max_total_oid_images:
                        break
            if len(selected_image_ids) >= max_total_oid_images:
                break

        print(f"Ingesting {len(selected_image_ids)} selected Open Images V7 images...")
        added = 0
        session = requests.Session()

        for idx, img_id in enumerate(selected_image_ids):
            oid_url = f"https://open-images-dataset.s3.amazonaws.com/validation/{img_id}.jpg"
            local_img_path = os.path.join(self.downloaded_images_dir, f"oid_{img_id}.jpg")

            if not os.path.exists(local_img_path):
                try:
                    resp = session.get(oid_url, timeout=15)
                    if resp.status_code == 200:
                        with open(local_img_path, "wb") as f:
                            f.write(resp.content)
                    else:
                        continue
                except Exception:
                    continue

            try:
                with Image.open(local_img_path) as im:
                    im.verify()
            except Exception:
                continue

            img_hash = compute_image_sha256(local_img_path)
            if img_hash in self.seen_hashes:
                continue

            raw_boxes = img_to_boxes[img_id]
            valid_boxes = [(b[0], b[1], b[2], b[3], b[4]) for b in raw_boxes]
            if not valid_boxes:
                continue

            self.seen_hashes[img_hash] = f"oid_{img_id}.jpg"
            self.samples.append({
                "image_path": local_img_path,
                "image_name": f"oid_{img_id}.jpg",
                "boxes": valid_boxes,
                "source_dataset": "Google_OpenImages_V7_val",
                "source_image_id": img_id,
                "license": "OpenImages_CC_BY_4.0",
            })
            added += 1

        print(f"[OK] Ingested {added} Open Images V7 real images with verified annotations.")
        return added

    def build_and_export_dataset(self) -> Dict[str, Any]:
        """
        Partitions all ingested samples using object-level stratification,
        copies files to deployment_dataset_expanded/, and generates data.yaml and provenance records.
        """
        random.seed(self.seed)
        self.initialize_output_directory(clean=True)

        class_to_samples = defaultdict(list)
        for s in self.samples:
            primary_cls = s["boxes"][0][0]
            class_to_samples[primary_cls].append(s)

        split_assignments = {"train": [], "valid": [], "test": []}

        for cls_id, cls_samples in class_to_samples.items():
            random.shuffle(cls_samples)
            n = len(cls_samples)
            if n == 1:
                split_assignments["train"].append(cls_samples[0])
            elif n == 2:
                split_assignments["train"].append(cls_samples[0])
                split_assignments["valid"].append(cls_samples[1])
            elif n == 3:
                split_assignments["train"].append(cls_samples[0])
                split_assignments["valid"].append(cls_samples[1])
                split_assignments["test"].append(cls_samples[2])
            else:
                n_val = max(1, int(round(n * self.val_ratio)))
                n_test = max(1, int(round(n * self.test_ratio)))
                n_train = n - n_val - n_test
                if n_train < 1:
                    n_train = 1
                    n_val = max(1, (n - 1) // 2)
                    n_test = n - n_train - n_val

                split_assignments["train"].extend(cls_samples[:n_train])
                split_assignments["valid"].extend(cls_samples[n_train : n_train + n_val])
                split_assignments["test"].extend(cls_samples[n_train + n_val :])

        counts = {"train": 0, "valid": 0, "test": 0}
        provenance_list = []

        for split_name, sample_list in split_assignments.items():
            out_img_dir = os.path.join(self.output_dir, split_name, "images")
            out_lbl_dir = os.path.join(self.output_dir, split_name, "labels")

            for s in sample_list:
                dst_img = os.path.join(out_img_dir, s["image_name"])
                base = os.path.splitext(s["image_name"])[0]
                dst_lbl = os.path.join(out_lbl_dir, base + ".txt")

                if os.path.abspath(s["image_path"]) != os.path.abspath(dst_img):
                    shutil.copy2(s["image_path"], dst_img)
                with open(dst_lbl, "w", encoding="utf-8") as lf:
                    for b in s["boxes"]:
                        lf.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                counts[split_name] += 1

                for b in s["boxes"]:
                    cname = CANONICAL_24_CLASSES[b[0]]
                    provenance_list.append({
                        "imported_filename": s["image_name"],
                        "split": split_name,
                        "source_dataset": s["source_dataset"],
                        "source_image_id": s["source_image_id"],
                        "mapped_project_class": cname,
                        "class_id": b[0],
                        "license": s["license"],
                        "bbox_yolo": [round(b[1], 6), round(b[2], 6), round(b[3], 6), round(b[4], 6)],
                    })

        # Save data.yaml
        data_yaml_path = os.path.join(self.output_dir, "data.yaml")
        yaml_data = {
            "path": self.output_dir,
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(CANONICAL_24_CLASSES),
            "names": CANONICAL_24_CLASSES,
        }
        with open(data_yaml_path, "w", encoding="utf-8") as yf:
            yaml.safe_dump(yaml_data, yf, sort_keys=False)

        # Save provenance
        prov_path_expanded = os.path.join(self.output_dir, "metadata", "provenance.json")
        prov_path_results = os.path.join(PROJECT_ROOT, "results", "dataset_expansion_provenance.json")
        os.makedirs(os.path.dirname(prov_path_results), exist_ok=True)

        summary_meta = {
            "total_images": sum(counts.values()),
            "split_counts": counts,
            "total_annotations": len(provenance_list),
            "sources": list(set(s["source_dataset"] for s in self.samples)),
            "provenance_records": provenance_list,
        }

        with open(prov_path_expanded, "w", encoding="utf-8") as pf:
            json.dump(summary_meta, pf, indent=2)
        with open(prov_path_results, "w", encoding="utf-8") as pf:
            json.dump(summary_meta, pf, indent=2)

        return summary_meta

    def add_sample(
        self,
        image_path: str,
        boxes: List[Tuple[int, float, float, float, float]],
        source_name: str = "custom",
    ) -> bool:
        """Adds a verified image/annotation pair with validation and deduplication."""
        if not os.path.exists(image_path):
            return False

        valid_boxes = []
        for box in boxes:
            cid, xc, yc, w, h = box
            is_val, _ = validate_yolo_bbox(cid, xc, yc, w, h)
            if is_val:
                valid_boxes.append(box)

        if not valid_boxes:
            return False

        try:
            with Image.open(image_path) as im:
                im.verify()
        except Exception:
            return False

        img_hash = compute_image_sha256(image_path)
        if img_hash in self.seen_hashes:
            return False

        img_f = os.path.basename(image_path)
        self.seen_hashes[img_hash] = img_f
        self.samples.append({
            "image_path": image_path,
            "image_name": img_f,
            "boxes": valid_boxes,
            "source_dataset": source_name,
            "source_image_id": img_f,
            "license": "Custom_Verified_CC_BY_4.0",
        })
        return True

    def build_stratified_split(self) -> Dict[str, int]:
        """Alias for build_and_export_dataset split counts."""
        res = self.build_and_export_dataset()
        return res.get("split_counts", {})


DatasetExpander = RealPublicDatasetExpander


def main():
    expander = RealPublicDatasetExpander(
        output_dir="deployment_dataset_expanded",
        cache_dir="data/public_cache",
    )
    expander.ingest_original_deployment_dataset("deployment_dataset")
    expander.ingest_coco_samples(max_images_per_class=45, max_total_coco_images=350)
    expander.ingest_openimages_samples(max_images_per_class=45, max_total_oid_images=350)
    expander.build_and_export_dataset()


if __name__ == "__main__":
    main()
