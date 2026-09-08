"""
train_cargo_yolo_exp_v3.py - High-Resolution Multi-Scale YOLOv8 Training Pipeline (Exp v3)
==========================================================================================
Project: Cargo Vision Logistics System (BUG #5 - Exp v3)
Purpose: Configures and executes controlled YOLOv8 training on deployment_dataset_expanded/
         incorporating high-resolution (800x800 px), multi-scale scale augmentation (0.5),
         increased box loss gain (8.5), and cosine AdamW scheduling.

Key Design Principles:
  - Isolated output checkpoint: models/cargo_yolo_exp_v3_best.pt (strictly preserves exp_v2 & 24class)
  - Dataset safety: uses deployment_dataset_expanded/ (train: 859, valid: 185, test: 185)
  - Programmatic data split verification: ensures 0 test-set overlap
  - Dry-run validation support (--dry-run) without starting training
  - Clean evaluation on held-out test split post-training
"""

import os
import sys
import shutil
import json
import argparse
import hashlib
from typing import Dict, Any, Tuple
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
import time

# Windows/OneDrive file lock mitigation for results.csv
_orig_save_metrics = BaseTrainer.save_metrics

def _robust_save_metrics(self, metrics=None):
    for attempt in range(10):
        try:
            return _orig_save_metrics(self, metrics=metrics)
        except PermissionError as e:
            if attempt < 9:
                time.sleep(1.0)
            else:
                print(f"[WARNING] save_metrics hit PermissionError on results.csv after 10 retries: {e}", flush=True)
                return

BaseTrainer.save_metrics = _robust_save_metrics


def compute_file_hash(filepath: str) -> str:
    """Computes MD5 hash of a file for duplicate/leakage detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_dataset_safety(data_yaml_path: str) -> Dict[str, Any]:
    """
    Programmatically verifies dataset integrity, split paths, and ensures 0 test-set leakage.
    """
    with open(data_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    yaml_dir = os.path.dirname(data_yaml_path)
    train_rel = yaml_data.get("train", "")
    valid_rel = yaml_data.get("val", yaml_data.get("valid", ""))
    test_rel = yaml_data.get("test", "")

    train_dir = os.path.abspath(os.path.join(yaml_dir, train_rel))
    valid_dir = os.path.abspath(os.path.join(yaml_dir, valid_rel))
    test_dir = os.path.abspath(os.path.join(yaml_dir, test_rel))

    for name, p in [("train", train_dir), ("valid", valid_dir), ("test", test_dir)]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Dataset split directory '{name}' not found at: {p}")

    # Gather image hashes per split
    split_hashes = {}
    split_counts = {}
    for name, p in [("train", train_dir), ("valid", valid_dir), ("test", test_dir)]:
        hashes = set()
        img_files = [f for f in os.listdir(p) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        split_counts[name] = len(img_files)
        for img in img_files:
            h = compute_file_hash(os.path.join(p, img))
            hashes.add(h)
        split_hashes[name] = hashes

    # Verify 0 overlap between splits
    train_test_overlap = split_hashes["train"].intersection(split_hashes["test"])
    valid_test_overlap = split_hashes["valid"].intersection(split_hashes["test"])
    train_valid_overlap = split_hashes["train"].intersection(split_hashes["valid"])

    if train_test_overlap:
        raise ValueError(f"CRITICAL: Found {len(train_test_overlap)} image hash collisions between TRAIN and TEST!")
    if valid_test_overlap:
        raise ValueError(f"CRITICAL: Found {len(valid_test_overlap)} image hash collisions between VALID and TEST!")

    return {
        "status": "VERIFIED_SAFE",
        "train_images": split_counts["train"],
        "valid_images": split_counts["valid"],
        "test_images": split_counts["test"],
        "train_test_overlap": len(train_test_overlap),
        "valid_test_overlap": len(valid_test_overlap),
        "train_valid_overlap": len(train_valid_overlap),
        "classes_count": yaml_data.get("nc", len(yaml_data.get("names", []))),
        "class_names": yaml_data.get("names", []),
    }


def validate_training_config(
    data_yaml: str = "deployment_dataset_expanded/data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 60,
    imgsz: int = 800,
    batch_size: int = 16,
    lr0: float = 0.001,
    lrf: float = 0.01,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 4.0,
    close_mosaic: int = 10,
    box: float = 8.5,
    scale: float = 0.5,
    mosaic: float = 1.0,
    seed: int = 42,
    project_dir: str = "runs/detect",
    name: str = "cargo_yolo_exp_v3",
    output_model_path: str = "models/cargo_yolo_exp_v3_best.pt",
) -> Dict[str, Any]:
    """
    Validates the complete Exp v3 training configuration without executing training.
    """
    data_yaml_path = os.path.abspath(os.path.join(PROJECT_ROOT, data_yaml))
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    base_model_path = os.path.abspath(os.path.join(PROJECT_ROOT, base_model))
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model checkpoint not found at: {base_model_path}")

    safety_report = verify_dataset_safety(data_yaml_path)

    config_summary = {
        "experiment_name": name,
        "base_model": base_model,
        "base_model_path": base_model_path,
        "dataset_yaml": data_yaml_path,
        "target_checkpoint": os.path.abspath(os.path.join(PROJECT_ROOT, output_model_path)),
        "run_directory": os.path.abspath(os.path.join(PROJECT_ROOT, project_dir, name)),
        "hyperparameters": {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "optimizer": "AdamW",
            "lr0": lr0,
            "lrf": lrf,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "cos_lr": True,
            "box_loss_gain": box,
            "scale_augmentation": scale,
            "mosaic_augmentation": mosaic,
            "close_mosaic_epochs": close_mosaic,
            "seed": seed,
            "deterministic": True,
            "workers": 0,
        },
        "dataset_safety": safety_report,
    }

    return config_summary


def train_cargo_yolo_exp_v3(
    data_yaml: str = "deployment_dataset_expanded/data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 60,
    imgsz: int = 800,
    batch_size: int = 16,
    lr0: float = 0.001,
    lrf: float = 0.01,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 4.0,
    close_mosaic: int = 10,
    box: float = 8.5,
    scale: float = 0.5,
    mosaic: float = 1.0,
    seed: int = 42,
    project_dir: str = "runs/detect",
    name: str = "cargo_yolo_exp_v3",
    output_model_path: str = "models/cargo_yolo_exp_v3_best.pt",
    resume: bool = False,
    eval_test: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Executes or dry-runs the High-Resolution Multi-Scale YOLOv8 training (Exp v3).
    """
    config = validate_training_config(
        data_yaml=data_yaml,
        base_model=base_model,
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size,
        lr0=lr0,
        lrf=lrf,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        close_mosaic=close_mosaic,
        box=box,
        scale=scale,
        mosaic=mosaic,
        seed=seed,
        project_dir=project_dir,
        name=name,
        output_model_path=output_model_path,
    )

    print("=" * 80)
    print("CARGO VISION - YOLOv8 HIGH-RESOLUTION TRAINING PIPELINE (EXP V3)")
    print("=" * 80)
    print(f"Experiment Name:       {config['experiment_name']}")
    print(f"Base Pretrained Model: {config['base_model']}")
    print(f"Dataset YAML:          {config['dataset_yaml']}")
    print(f"Target Checkpoint:     {config['target_checkpoint']}")
    print(f"Image Resolution:      {imgsz}x{imgsz} px (High-Res)")
    print(f"Target Epochs:         {epochs}")
    print(f"Batch Size:            {batch_size}")
    print(f"Optimizer:             AdamW (lr0={lr0}, lrf={lrf}, cos_lr=True)")
    print(f"Box Loss Gain:         {box} (Up from default 7.5)")
    print(f"Scale Augmentation:    {scale} (Multi-scale range +/- 50%)")
    print(f"Mosaic / Close Mosaic: {mosaic} / {close_mosaic} epochs")
    print(f"Random Seed:           {seed} (deterministic=True)")
    print(f"Resume Mode:           {resume}")
    print(f"Dataset Verification:  TRAIN={config['dataset_safety']['train_images']} | VALID={config['dataset_safety']['valid_images']} | TEST={config['dataset_safety']['test_images']}")
    print(f"Test Set Isolation:    0 train/test overlap, 0 valid/test overlap (SAFE)")
    print("=" * 80)

    if dry_run:
        print("\n[DRY RUN MODE] Configuration and dataset safety validated successfully.")
        print("[DRY RUN MODE] Training was NOT started (as requested).")
        return {"status": "DRY_RUN_SUCCESS", "config": config}

    # Setup directories
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "training"), exist_ok=True)

    save_dir = os.path.join(project_dir, name)
    last_weights_path = os.path.join(save_dir, "weights", "last.pt")

    if resume:
        if not os.path.exists(last_weights_path):
            # check nested run directory if created by Ultralytics
            nested_last = os.path.join(project_dir, project_dir, name, "weights", "last.pt")
            if os.path.exists(nested_last):
                last_weights_path = nested_last
            else:
                raise FileNotFoundError(f"Cannot resume: last checkpoint not found at: {last_weights_path}")
        print(f"\n[RESUME] Resuming training from checkpoint: {last_weights_path}")
        model = YOLO(last_weights_path)
        train_results = model.train(resume=True)
    else:
        # 1. Initialize YOLO model
        model = YOLO(config["base_model_path"])

        # 2. Execute Training
        train_results = model.train(
            data=config["dataset_yaml"],
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=lr0,
            lrf=lrf,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            optimizer="AdamW",
            cos_lr=True,
            box=box,
            scale=scale,
            mosaic=mosaic,
            close_mosaic=close_mosaic,
            seed=seed,
            deterministic=True,
            # Execution & Saving
            project=project_dir,
            name=name,
            exist_ok=True,
            verbose=True,
            val=True,
            save=True,
            plots=True,
            workers=0,
        )


    # 3. Locate best model weights
    save_dir = str(getattr(train_results, "save_dir", os.path.join(project_dir, name)))
    best_weights_path = os.path.join(save_dir, "weights", "best.pt")
    last_weights_path = os.path.join(save_dir, "weights", "last.pt")

    target_path = config["target_checkpoint"]
    if os.path.exists(best_weights_path):
        shutil.copy2(best_weights_path, target_path)
        print(f"\n[OK] Copied best weights to: {target_path}")
    elif os.path.exists(last_weights_path):
        shutil.copy2(last_weights_path, target_path)
        print(f"\n[OK] Copied last weights to: {target_path}")
    else:
        raise FileNotFoundError(f"Could not locate trained weights in: {save_dir}")

    # 4. Post-training validation on VALID split
    print("\n" + "=" * 80)
    print("EVALUATING BEST EXP V3 CHECKPOINT ON VALIDATION SPLIT...")
    print("=" * 80)
    best_model = YOLO(target_path)
    val_metrics = best_model.val(
        data=config["dataset_yaml"],
        split="val",
        imgsz=imgsz,
        batch=batch_size,
        verbose=True,
        workers=0,
    )

    eval_summary = {
        "experiment": name,
        "checkpoint": target_path,
        "imgsz": imgsz,
        "epochs": epochs,
        "val_precision": round(float(val_metrics.results_dict.get("metrics/precision(B)", 0.0)), 4),
        "val_recall": round(float(val_metrics.results_dict.get("metrics/recall(B)", 0.0)), 4),
        "val_mAP50": round(float(val_metrics.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        "val_mAP50_95": round(float(val_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
    }

    if eval_test:
        print("\n" + "=" * 80)
        print("EVALUATING EXP V3 CHECKPOINT ON HELD-OUT TEST SPLIT...")
        print("=" * 80)
        test_metrics = best_model.val(
            data=config["dataset_yaml"],
            split="test",
            imgsz=imgsz,
            batch=batch_size,
            verbose=True,
            workers=0,
        )
        eval_summary.update({
            "test_precision": round(float(test_metrics.results_dict.get("metrics/precision(B)", 0.0)), 4),
            "test_recall": round(float(test_metrics.results_dict.get("metrics/recall(B)", 0.0)), 4),
            "test_mAP50": round(float(test_metrics.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
            "test_mAP50_95": round(float(test_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
        })

    eval_json_path = os.path.join(PROJECT_ROOT, "results", "training", f"{name}_val_report.json")
    with open(eval_json_path, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"\n[OK] Validation report saved to: {eval_json_path}")

    return {"status": "TRAINING_COMPLETED", "config": config, "metrics": eval_summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cargo Vision YOLOv8 Exp v3 Training Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset safety and configuration without training")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt checkpoint")
    parser.add_argument("--epochs", type=int, default=60, help="Target training epochs (default: 60)")
    parser.add_argument("--imgsz", type=int, default=800, help="Image resolution in pixels (default: 800)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--eval-test", action="store_true", help="Evaluate on held-out test split post-training")
    args = parser.parse_args()

    train_cargo_yolo_exp_v3(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        resume=args.resume,
        eval_test=args.eval_test,
        dry_run=args.dry_run,
    )
