"""
integrate_supplemental_dataset.py - Non-Destructive Dataset Integration & Stratification Engine
=============================================================================================
Project: Cargo Vision Logistics System
Purpose: Ingests all 949 base expanded samples (249 baseline + 350 COCO + 350 OpenImages Batch 1)
         and verifies, deduplicates, and integrates all 300 supplemental Open Images V7 samples
         from data/openimages_supplemental/ into deployment_dataset_expanded/.
         Guarantees 100% preservation of deployment_dataset/, 0 duplicate leakage, and full provenance.
"""

import os
import sys
import yaml
import json
import shutil
import hashlib
import random
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.expand_dataset import (
    CANONICAL_24_CLASSES,
    CLASS_NAME_TO_ID,
    RealPublicDatasetExpander,
    validate_yolo_bbox,
    compute_image_sha256,
)


class SupplementalDatasetIntegrator:
    """
    Integrates verified supplemental Open Images samples into deployment_dataset_expanded.
    """

    def __init__(
        self,
        output_dir: str = "deployment_dataset_expanded",
        supplement_dir: str = "data/openimages_supplemental",
        cache_dir: str = "data/public_cache",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.supplement_dir = os.path.abspath(supplement_dir)
        self.cache_dir = os.path.abspath(cache_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.expander = RealPublicDatasetExpander(
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed,
        )

    def run_integration(self) -> Dict[str, Any]:
        """
        Executes full non-destructive integration of:
        1. Original deployment_dataset (249 valid images)
        2. MS COCO 2017 (350 images)
        3. Google Open Images V7 Batch 1 (350 images)
        4. Google Open Images V7 Supplemental Batch (300 images)
        Total Target: 1,249 images across all 24 classes.
        """
        print("\n" + "=" * 80)
        print("CARGO VISION: COMPREHENSIVE DATASET INTEGRATION ENGINE")
        print("=" * 80)

        # Step 1: Base expansion ingestion
        print("[STEP 1/4] Ingesting original deployment_dataset/ (249 valid images)...")
        self.expander.ingest_original_deployment_dataset("deployment_dataset")

        print("[STEP 2/4] Ingesting MS COCO 2017 validation samples (350 images)...")
        self.expander.ingest_coco_samples(max_total_coco_images=350)

        print("[STEP 3/4] Ingesting Google Open Images V7 Batch 1 samples (350 images)...")
        self.expander.ingest_openimages_samples(max_total_oid_images=350)

        base_samples_count = len(self.expander.samples)
        print(f"[OK] Base dataset assembled: {base_samples_count} images.")

        # Step 4: Supplemental OpenImages Ingestion
        print(f"[STEP 4/4] Ingesting supplemental Open Images V7 samples from {self.supplement_dir}...")
        supp_img_dir = os.path.join(self.supplement_dir, "images")
        supp_lbl_dir = os.path.join(self.supplement_dir, "labels")

        if not os.path.exists(supp_img_dir) or not os.path.exists(supp_lbl_dir):
            raise FileNotFoundError(f"Supplemental directory missing: {self.supplement_dir}")

        supp_files = sorted([f for f in os.listdir(supp_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        supp_added = 0
        supp_boxes = 0

        for img_f in supp_files:
            base = os.path.splitext(img_f)[0]
            lbl_f = base + ".txt"
            img_p = os.path.join(supp_img_dir, img_f)
            lbl_p = os.path.join(supp_lbl_dir, lbl_f)

            if not os.path.exists(lbl_p):
                continue

            try:
                with Image.open(img_p) as im:
                    im.verify()
            except Exception:
                continue

            img_hash = compute_image_sha256(img_p)
            if img_hash in self.expander.seen_hashes:
                continue

            valid_boxes = []
            with open(lbl_p, "r", encoding="utf-8") as lf:
                lines = [l.strip() for l in lf.readlines() if l.strip()]

            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    is_val, _ = validate_yolo_bbox(cid, xc, yc, w, h)
                    if is_val:
                        valid_boxes.append((cid, xc, yc, w, h))

            if not valid_boxes:
                continue

            self.expander.seen_hashes[img_hash] = img_f
            self.expander.samples.append({
                "image_path": img_p,
                "image_name": img_f,
                "boxes": valid_boxes,
                "source_dataset": "Google_OpenImages_V7_supplemental",
                "source_image_id": img_f,
                "license": "OpenImages_CC_BY_4.0",
            })
            supp_added += 1
            supp_boxes += len(valid_boxes)

        print(f"[OK] Successfully verified and integrated {supp_added} supplemental images ({supp_boxes} boxes).")
        print(f"[TOTAL IMAGES READY FOR STRATIFICATION]: {len(self.expander.samples)}")

        # Build final stratified split
        summary_meta = self.expander.build_and_export_dataset()
        return summary_meta


def main():
    integrator = SupplementalDatasetIntegrator(
        output_dir="deployment_dataset_expanded",
        supplement_dir="data/openimages_supplemental",
        cache_dir="data/public_cache",
    )
    integrator.run_integration()


if __name__ == "__main__":
    main()
