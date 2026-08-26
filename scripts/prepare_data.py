"""
prepare_data.py - Open Images Data Sourcing, Filtering, and Partitioning Pipeline
==================================================================================
Project: Cargo Vision CNN
Purpose: Downloads, filters, verifies, and splits a high-quality subset of Google
         Open Images for 5 cargo categories (box, suitcase, chair, table, sofa).

Target Classes and Open Images MID Mapping:
  - box:      /m/025dyy  (Box / shipping carton)
  - suitcase: /m/01s55n  (Suitcase / travel luggage)
  - chair:    /m/01mzpv  (Chair / seating furniture)
  - table:    /m/04bcr3  (Table / flat surface furniture)
  - sofa:     /m/02crq1  (Couch / upholstered sofa)

Quality & Filtering Rules:
  1. Single-class exclusivity: Rejects images with multiple competing target classes.
  2. Image integrity verification: Verifies all JPEGs with PIL.Image before writing.
  3. Deduplication: Ensures every image ID is unique across the entire dataset.
  4. Reproducible partitioning: 70% Train, 15% Validation, 15% Test using fixed seed.
  5. Provenance tracking: Saves full audit metadata to data/dataset_provenance.json.
"""

import os
import io
import json
import time
import shutil
import random
import argparse
from typing import Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
from PIL import Image

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

TARGET_CLASSES: Dict[str, str] = {
    "/m/025dyy": "box",
    "/m/01s55n": "suitcase",
    "/m/01mzpv": "chair",
    "/m/04bcr3": "table",
    "/m/02crq1": "sofa",
}

# Open Images Public S3 Base URLs
S3_BASE_URLS = {
    "validation": "https://open-images-dataset.s3.amazonaws.com/validation",
    "test": "https://open-images-dataset.s3.amazonaws.com/test",
    "train": "https://open-images-dataset.s3.amazonaws.com/train",
}

# Open Images Metadata URLs (Official Google Storage CSVs)
METADATA_URLS = {
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels.csv",
    "train": "https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels.csv",
}

DEFAULT_TARGET_PER_CLASS = 300
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_SEED = 42
DEFAULT_WORKERS = 8


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download, filter, verify, and split Open Images data for Cargo Vision CNN."
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=DEFAULT_TARGET_PER_CLASS,
        help=f"Target number of clean images per class (default: {DEFAULT_TARGET_PER_CLASS})",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory to save verified raw downloaded images (default: data/raw)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory to save partitioned train/validation/test sets (default: data/processed)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Training set split fraction (default: {DEFAULT_TRAIN_RATIO})",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f"Validation set split fraction (default: {DEFAULT_VAL_RATIO})",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"Test set split fraction (default: {DEFAULT_TEST_RATIO})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible shuffling and splitting (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of concurrent download worker threads (default: {DEFAULT_WORKERS})",
    )
    return parser.parse_args()


def fetch_candidate_image_ids(
    target_per_class: int,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Identifies clean candidate images from Open Images metadata.
    Enforces strict global single-class exclusivity across all processed metadata
    (validation, test, and complete train stream) to eliminate ambiguous multi-object samples.
    
    Split ID Guarantee:
        In Open Images, ImageIDs are 16-hex characters uniquely identifying individual Flickr photos.
        Validation, test, and train splits are strictly disjoint partitions.
        image_to_split records the split origin for each ImageID.
    
    Returns:
        Dict mapping class_name -> list of (image_id, split_name) tuples.
    """
    print("\n[Step 1/4] Querying Open Images metadata for clean candidate images...")
    target_mids = set(TARGET_CLASSES.keys())

    # Global tracking of target classes associated with each ImageID
    # Maps ImageID -> set of target class names positively annotated (Confidence == 1)
    image_to_target_classes: Dict[str, Set[str]] = {}
    # Maps ImageID -> Open Images split name ("validation", "test", or "train")
    image_to_split: Dict[str, str] = {}

    # 1. Ingest validation and test metadata
    for split in ["validation", "test"]:
        url = METADATA_URLS[split]
        print(f"  -> Ingesting {split} split image labels from Open Images...")
        try:
            df = pd.read_csv(url)
            # Filter for positive annotations (Confidence == 1) of our target classes
            df_target = df[(df["LabelName"].isin(target_mids)) & (df["Confidence"] == 1)]
            for _, row in df_target.iterrows():
                img_id = row["ImageID"]
                c = TARGET_CLASSES[row["LabelName"]]
                if img_id not in image_to_target_classes:
                    image_to_target_classes[img_id] = set()
                    image_to_split[img_id] = split
                image_to_target_classes[img_id].add(c)
        except Exception as e:
            print(f"  [Warning] Failed to fetch {url}: {e}")

    # Preliminary audit of pure candidates available from validation + test
    prelim_pure = {c: 0 for c in TARGET_CLASSES.values()}
    for img_id, classes in image_to_target_classes.items():
        if len(classes) == 1:
            prelim_pure[next(iter(classes))] += 1

    # 2. Check if any class requires train split supplementation
    # A buffer of target_per_class * 1.5 ensures enough candidate files even after
    # PIL image corruption and size filtering during download.
    shortfall = {
        c: int(target_per_class * 1.5) - count
        for c, count in prelim_pure.items()
        if count < int(target_per_class * 1.5)
    }

    if shortfall:
        print(f"  -> Supplementing rarer classes ({list(shortfall.keys())}) from complete train metadata stream...")
        try:
            train_url = METADATA_URLS["train"]
            # Stream the complete train metadata in chunks, accumulating all positive target labels globally.
            # No early break is used to guarantee that multi-label occurrences across different chunks
            # are fully captured before making any purity decisions.
            chunk_count = 0
            for chunk in pd.read_csv(train_url, chunksize=100000):
                chunk_count += 1
                sub = chunk[(chunk["LabelName"].isin(target_mids)) & (chunk["Confidence"] == 1)]
                for _, row in sub.iterrows():
                    img_id = row["ImageID"]
                    c = TARGET_CLASSES[row["LabelName"]]
                    if img_id not in image_to_target_classes:
                        image_to_target_classes[img_id] = set()
                        image_to_split[img_id] = "train"
                    image_to_target_classes[img_id].add(c)
            print(f"  -> Ingested train split stream ({chunk_count} chunks processed).")
        except Exception as e:
            print(f"  [Notice] Streamed train split: {e}")

    # 3. Post-collection Global Purity Decision
    # Evaluated ONLY AFTER all relevant metadata has been fully ingested.
    # An ImageID is accepted if and only if exactly 1 target class was annotated globally.
    candidates: Dict[str, List[Tuple[str, str]]] = {c: [] for c in TARGET_CLASSES.values()}
    disqualified_multi_class_count = 0

    for img_id, classes in image_to_target_classes.items():
        if len(classes) == 1:
            single_class = next(iter(classes))
            split = image_to_split[img_id]
            candidates[single_class].append((img_id, split))
        else:
            disqualified_multi_class_count += 1

    print(f"  -> Global purity filter complete: Disqualified {disqualified_multi_class_count} multi-target images.")
    for c in candidates:
        print(f"  Found {len(candidates[c])} globally-pure candidate IDs for class: '{c}'")

    return candidates


def download_and_verify_single_image(
    image_id: str,
    split: str,
    class_name: str,
    dest_path: str,
) -> bool:
    """
    Downloads a single image from AWS Open Data, validates it via PIL, and saves it.
    Returns True if successfully downloaded and verified, False otherwise.
    """
    if os.path.exists(dest_path):
        return True

    url = f"{S3_BASE_URLS[split]}/{image_id}.jpg"
    try:
        response = requests.get(url, timeout=12)
        if response.status_code != 200 or len(response.content) < 1024:
            return False

        # Verify image using Pillow
        image_bytes = io.BytesIO(response.content)
        with Image.open(image_bytes) as img:
            img.verify()  # Verifies file integrity and header

        # Re-open to convert and save as clean RGB JPEG
        image_bytes.seek(0)
        with Image.open(image_bytes) as img:
            rgb_img = img.convert("RGB")
            # Quality check: discard extremely tiny thumbnail images
            if rgb_img.width < 80 or rgb_img.height < 80:
                return False
            rgb_img.save(dest_path, "JPEG", quality=90)
        return True
    except Exception:
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        return False


def download_dataset_subset(
    candidates: Dict[str, List[Tuple[str, str]]],
    raw_dir: str,
    target_per_class: int,
    workers: int,
) -> Dict[str, List[str]]:
    """
    Downloads and verifies candidate images concurrently up to target_per_class per class.
    Returns Dict mapping class_name -> list of verified local image file paths.
    """
    print("\n[Step 2/4] Downloading and verifying image files...")
    verified_files: Dict[str, List[str]] = {c: [] for c in candidates}

    for class_name, img_tuples in candidates.items():
        class_dir = os.path.join(raw_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"\n  Downloading class '{class_name}' (Target: ~{target_per_class} images)...")

        needed = target_per_class
        download_tasks = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for img_id, split in img_tuples:
                dest_path = os.path.join(class_dir, f"{img_id}.jpg")
                task = executor.submit(
                    download_and_verify_single_image,
                    img_id,
                    split,
                    class_name,
                    dest_path,
                )
                download_tasks.append((task, dest_path))

            downloaded_count = 0
            for task, dest_path in download_tasks:
                success = task.result()
                if success:
                    verified_files[class_name].append(dest_path)
                    downloaded_count += 1
                    if downloaded_count >= needed:
                        break

        print(f"  -> Verified {len(verified_files[class_name])} valid images for '{class_name}'.")

    return verified_files


def partition_dataset(
    verified_files: Dict[str, List[str]],
    processed_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Dict[str, int]]:
    """
    Shuffles and splits verified images into train, validation, and test folders.
    Guarantees no duplicate images exist across splits.
    
    Returns:
        Nested dict with counts per split and class.
    """
    print("\n[Step 3/4] Partitioning dataset into train / validation / test splits...")
    random.seed(seed)

    splits = ["train", "validation", "test"]
    split_counts: Dict[str, Dict[str, int]] = {s: {c: 0 for c in verified_files} for s in splits}

    # Reset processed directory to ensure clean, isolated splits
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)

    for s in splits:
        for c in verified_files:
            os.makedirs(os.path.join(processed_dir, s, c), exist_ok=True)

    for class_name, file_paths in verified_files.items():
        # Deterministic shuffle with fixed seed
        shuffled = list(file_paths)
        random.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # Remainder goes to test set to ensure 100% of images are utilized
        n_test = n_total - n_train - n_val

        train_files = shuffled[:n_train]
        val_files = shuffled[n_train : n_train + n_val]
        test_files = shuffled[n_train + n_val :]

        # Copy files to their respective destination directories
        for f in train_files:
            shutil.copy2(f, os.path.join(processed_dir, "train", class_name, os.path.basename(f)))
            split_counts["train"][class_name] += 1

        for f in val_files:
            shutil.copy2(f, os.path.join(processed_dir, "validation", class_name, os.path.basename(f)))
            split_counts["validation"][class_name] += 1

        for f in test_files:
            shutil.copy2(f, os.path.join(processed_dir, "test", class_name, os.path.basename(f)))
            split_counts["test"][class_name] += 1

    return split_counts


def save_provenance_metadata(
    split_counts: Dict[str, Dict[str, int]],
    metadata_path: str,
    args: argparse.Namespace,
) -> None:
    """Saves dataset provenance and audit metadata to JSON."""
    provenance = {
        "dataset_name": "Google Open Images (Curated 5-Class Logistics Subset)",
        "source": "Open Images Dataset via AWS Open Data Registry & Google Cloud Storage",
        "provenance_url": "https://storage.googleapis.com/openimages/web/index.html",
        "license_annotations": "CC BY 4.0 (Creative Commons Attribution 4.0 International)",
        "license_images": "Creative Commons Attribution / CC-BY-SA / Public Domain",
        "classes": {
            "box": {"mid": "/m/025dyy", "description": "Box / shipping carton"},
            "suitcase": {"mid": "/m/01s55n", "description": "Suitcase / travel luggage"},
            "chair": {"mid": "/m/01mzpv", "description": "Chair / seating furniture"},
            "table": {"mid": "/m/04bcr3", "description": "Table / desk / flat furniture"},
            "sofa": {"mid": "/m/02crq1", "description": "Couch / upholstered sofa"},
        },
        "split_ratios": {
            "train": args.train_ratio,
            "validation": args.val_ratio,
            "test": args.test_ratio,
        },
        "random_seed": args.seed,
        "target_per_class": args.target_per_class,
        "counts": split_counts,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"\n[Step 4/4] Saved dataset provenance metadata to '{metadata_path}'.")


def print_summary_table(split_counts: Dict[str, Dict[str, int]], target_per_class: int) -> bool:
    """
    Prints a clean ASCII summary table of class counts across splits.
    Audits actual counts against the requested target_per_class.
    Returns True if all classes met or exceeded target, False otherwise.
    """
    classes = list(TARGET_CLASSES.values())
    print("\n" + "=" * 70)
    print(f"{'Class Name':<12} | {'Train (70%)':<12} | {'Val (15%)':<12} | {'Test (15%)':<12} | {'Total':<8} | {'Status':<10}")
    print("-" * 70)

    tot_train, tot_val, tot_test = 0, 0, 0
    all_targets_met = True
    shortfalls = []

    for c in classes:
        tr = split_counts["train"][c]
        va = split_counts["validation"][c]
        te = split_counts["test"][c]
        tot = tr + va + te
        tot_train += tr
        tot_val += va
        tot_test += te

        if tot >= target_per_class:
            status_str = "Met Target"
        else:
            status_str = f"{tot}/{target_per_class}"
            all_targets_met = False
            shortfalls.append((c, tot, target_per_class))

        print(f"{c:<12} | {tr:<12} | {va:<12} | {te:<12} | {tot:<8} | {status_str:<10}")

    print("-" * 70)
    grand_total = tot_train + tot_val + tot_test
    print(
        f"{'TOTAL':<12} | {tot_train:<12} | {tot_val:<12} | {tot_test:<12} | {grand_total:<8} |"
    )
    print("=" * 70)

    if shortfalls:
        print("\n[Audit Notice] The following classes yielded fewer samples than the requested target:")
        for c, count, target in shortfalls:
            print(f"  - '{c}': {count} clean verified samples obtained (requested target: {target}).")
        print("  -> Quality & label-purity filtering prioritized over raw sample count.")
    else:
        print("\n[Audit Notice] All classes successfully reached or exceeded the requested target count.")

    return all_targets_met


# -----------------------------------------------------------------------------
# Main Execution Entrypoint
# -----------------------------------------------------------------------------

def main():
    args = parse_arguments()
    print("=" * 70)
    print("Cargo Vision CNN - Dataset Preparation Pipeline")
    print(f"Target: ~{args.target_per_class} images/class | Seed: {args.seed} | Workers: {args.workers}")
    print("=" * 70)

    # 1. Fetch candidate image IDs with single-class exclusivity
    candidates = fetch_candidate_image_ids(target_per_class=args.target_per_class)

    # 2. Download and verify images
    verified_files = download_dataset_subset(
        candidates=candidates,
        raw_dir=args.raw_dir,
        target_per_class=args.target_per_class,
        workers=args.workers,
    )

    # 3. Partition into Train (70%), Validation (15%), Test (15%)
    split_counts = partition_dataset(
        verified_files=verified_files,
        processed_dir=args.processed_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # 4. Save metadata & provenance
    save_provenance_metadata(
        split_counts=split_counts,
        metadata_path="data/dataset_provenance.json",
        args=args,
    )

    # 5. Output summary table with target count audit
    all_targets_met = print_summary_table(split_counts, target_per_class=args.target_per_class)
    if all_targets_met:
        print("\n[STATUS] Dataset preparation completed successfully (all targets met).")
    else:
        print("\n[STATUS] Dataset preparation completed with filtered counts (cleanliness prioritized).")


if __name__ == "__main__":
    main()

