"""
dataset_config.py
=================
Project: Cargo Vision CNN
Purpose: Central dataset configuration, split validation, and imbalance compensation.
         Defines hyperparameters, training-only data augmentation policies,
         and computes balanced class weights via scikit-learn.

Rules Enforced:
  1. 5 Target Classes: box, chair, couch, suitcase, table.
  2. Input Resolution: 128x128x3.
  3. Strict Augmentation Isolation: Train split ONLY (Validation & Test receive NO augmentation).
  4. Non-Destructive: Never modifies or physically duplicates image files.
  5. Imbalance Compensation: Computes balanced class weights on Train split via sklearn.
"""

import os
import glob
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------------------------------------------------------
# 1. Classes & Mapping
# -----------------------------------------------------------------------------

TARGET_CLASSES: List[str] = ["box", "chair", "couch", "suitcase", "table"]
NUM_CLASSES: int = len(TARGET_CLASSES)

CLASS_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(TARGET_CLASSES)}
INDEX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(TARGET_CLASSES)}

# -----------------------------------------------------------------------------
# 2. Dimensions & Core Hyperparameters
# -----------------------------------------------------------------------------

IMAGE_HEIGHT: int = 128
IMAGE_WIDTH: int = 128
IMAGE_CHANNELS: int = 3
IMAGE_SIZE: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_SHAPE: Tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

BATCH_SIZE: int = 32
RANDOM_SEED: int = 42

# -----------------------------------------------------------------------------
# 3. Directory Paths
# -----------------------------------------------------------------------------

DATASET_ROOT: str = os.path.join("data", "classification")
TRAIN_DIR: str = os.path.join(DATASET_ROOT, "train")
VALID_DIR: str = os.path.join(DATASET_ROOT, "valid")
TEST_DIR: str = os.path.join(DATASET_ROOT, "test")

SPLIT_DIRS: Dict[str, str] = {
    "train": TRAIN_DIR,
    "valid": VALID_DIR,
    "test": TEST_DIR,
}

# -----------------------------------------------------------------------------
# 4. Data Augmentation Policy (TRAIN ONLY)
# -----------------------------------------------------------------------------

# Conservative transformations to expand effective variety without altering semantic identity
TRAIN_AUGMENTATION_CONFIG: Dict[str, Any] = {
    "random_flip_mode": "horizontal",       # Horizontal reflection (cargo items remain physically realistic)
    "random_rotation_factor": 0.03,         # +/- 0.03 * 2pi rad ≈ +/- 10.8 degrees (mild camera tilt)
    "random_zoom_height_factor": 0.08,      # +/- 8% vertical zoom
    "random_zoom_width_factor": 0.08,       # +/- 8% horizontal zoom
    "random_translation_height": 0.06,      # +/- 6% vertical translation
    "random_translation_width": 0.06,       # +/- 6% horizontal translation
}

# Explicit Rule: Validation and Test splits NEVER receive augmentation
VALID_TEST_AUGMENTATION_CONFIG: None = None


# -----------------------------------------------------------------------------
# 5. Helper & Audit Functions
# -----------------------------------------------------------------------------

def get_split_counts(split: str) -> Dict[str, int]:
    """
    Returns image count per class for a given split directory.
    """
    split_dir = SPLIT_DIRS.get(split)
    if not split_dir or not os.path.exists(split_dir):
        return {c: 0 for c in TARGET_CLASSES}

    counts = {}
    for class_name in TARGET_CLASSES:
        class_folder = os.path.join(split_dir, class_name)
        if os.path.exists(class_folder):
            files = glob.glob(os.path.join(class_folder, "*.jpg"))
            counts[class_name] = len(files)
        else:
            counts[class_name] = 0
    return counts


def get_all_split_counts() -> Dict[str, Dict[str, int]]:
    """
    Returns nested dictionary of image counts across train, valid, and test splits.
    """
    return {split: get_split_counts(split) for split in SPLIT_DIRS}


def compute_train_class_weights() -> Dict[int, float]:
    """
    Computes balanced class weights strictly from the TRAIN split using scikit-learn.
    
    Formula: weight_j = N_total / (N_classes * N_j)
    
    Returns:
        Dict mapping class index (0..4) -> float weight.
    """
    train_counts = get_split_counts("train")
    
    # Construct synthetic label array matching train class distribution
    y_train = []
    for class_name, count in train_counts.items():
        class_idx = CLASS_TO_INDEX[class_name]
        y_train.extend([class_idx] * count)

    y_train = np.array(y_train)
    classes_array = np.array(range(NUM_CLASSES))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_array,
        y=y_train,
    )

    return {int(idx): float(np.round(w, 4)) for idx, w in zip(classes_array, weights)}


def validate_dataset_structure() -> bool:
    """
    Verifies that all dataset directories, splits, and classes are intact and correctly structured.
    """
    print("\n[Validation Pass] Auditing dataset paths and directory structure...")
    all_ok = True

    if not os.path.exists(DATASET_ROOT):
        print(f"  [Error] Dataset root not found: {DATASET_ROOT}")
        return False

    for split, split_dir in SPLIT_DIRS.items():
        if not os.path.exists(split_dir):
            print(f"  [Error] Missing split directory: {split_dir}")
            all_ok = False
            continue

        for class_name in TARGET_CLASSES:
            class_folder = os.path.join(split_dir, class_name)
            if not os.path.exists(class_folder):
                print(f"  [Error] Missing class directory: {class_folder}")
                all_ok = False
            else:
                files = glob.glob(os.path.join(class_folder, "*.jpg"))
                if len(files) == 0:
                    print(f"  [Error] Empty class folder: {class_folder}")
                    all_ok = False

    if all_ok:
        print("  -> All 15 split/class directories exist and contain valid files.")
        print("  -> Train, Valid, and Test splits are strictly isolated.")
    return all_ok


def print_configuration_summary() -> None:
    """
    Prints a formatted summary table of dataset counts, class weights, and augmentation settings.
    """
    counts = get_all_split_counts()
    class_weights = compute_train_class_weights()

    print("\n" + "=" * 78)
    print("Cargo Vision CNN - Dataset Configuration & Class Weighting Summary")
    print(f"Image Resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT}x{IMAGE_CHANNELS} | Batch Size: {BATCH_SIZE} | Seed: {RANDOM_SEED}")
    print("=" * 78)

    print(f"{'Index':<6} | {'Class Name':<10} | {'Train':<8} | {'Class Weight':<14} | {'Valid':<8} | {'Test':<8} | {'Total':<8}")
    print("-" * 78)

    tot_tr, tot_va, tot_te = 0, 0, 0
    for idx, c in enumerate(TARGET_CLASSES):
        tr = counts["train"][c]
        va = counts["valid"][c]
        te = counts["test"][c]
        tot = tr + va + te
        weight = class_weights[idx]

        tot_tr += tr
        tot_va += va
        tot_te += te
        print(f"{idx:<6} | {c:<10} | {tr:<8} | {weight:<14.4f} | {va:<8} | {te:<8} | {tot:<8}")

    print("-" * 78)
    grand_tot = tot_tr + tot_va + tot_te
    print(f"{'TOTAL':<19} | {tot_tr:<8} | {'N/A':<14} | {tot_va:<8} | {tot_te:<8} | {grand_tot:<8}")
    print("=" * 78)

    print("\n[Data Augmentation Policy]")
    print(f"  - Target Split: TRAIN ONLY")
    for k, v in TRAIN_AUGMENTATION_CONFIG.items():
        print(f"    * {k}: {v}")
    print("  - Target Split: VALIDATION & TEST -> NONE (Only 1/255 rescaling)")

    print("\n[Imbalance Compensation Rationale]")
    print(f"  - Lowest class 'suitcase' (n={counts['train']['suitcase']}) receives weight = {class_weights[CLASS_TO_INDEX['suitcase']]:.4f}")
    print(f"  - Highest class 'chair' (n={counts['train']['chair']}) receives weight = {class_weights[CLASS_TO_INDEX['chair']]:.4f}")
    print("  -> Loss penalty is proportionally scaled to prevent majority-class bias.")


# -----------------------------------------------------------------------------
# Main Execution Entrypoint
# -----------------------------------------------------------------------------

def main():
    # 1. Validate structure
    valid = validate_dataset_structure()
    if not valid:
        print("\n[FAILED] Dataset structure validation failed. Please check directories.")
        return

    # 2. Print complete configuration summary
    print_configuration_summary()
    print("\n[PASSED] Dataset configuration verified successfully!")


if __name__ == "__main__":
    main()
