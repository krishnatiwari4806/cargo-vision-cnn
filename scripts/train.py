"""
train.py - Training and Evaluation Pipeline for Cargo Vision CNN
================================================================
Project: Cargo Vision CNN
Purpose: Trains the custom from-scratch CNN model on the 5-class cargo dataset,
         applies training-only data augmentation, enforces balanced class weights,
         logs training progress, and generates diagnostic evaluation metrics.

Workflow:
  1. Loads dataset splits (data/classification/{train, valid, test}).
  2. Applies conservative on-the-fly data augmentation to TRAIN split only.
  3. Computes balanced class weights on Train split to compensate for imbalance.
  4. Builds and compiles the custom CNN architecture from scripts/model.py.
  5. Trains with EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, and CSVLogger.
  6. Evaluates the best model on the untouched test split.
  7. Generates training loss/accuracy curves, confusion matrix, and classification report.
  8. Saves trained model artifacts in models/ and diagnostic plots in results/training/.
"""

import os
import sys
import json
import time
import argparse
from typing import Tuple, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

# Ensure project root is in sys.path regardless of execution CWD
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Central project modules
try:
    from scripts.dataset_config import (
        TARGET_CLASSES,
        NUM_CLASSES,
        IMAGE_SIZE,
        BATCH_SIZE,
        RANDOM_SEED,
        TRAIN_DIR,
        VALID_DIR,
        TEST_DIR,
        compute_train_class_weights,
    )
    from scripts.model import build_cargo_cnn
except ModuleNotFoundError:
    from dataset_config import (
        TARGET_CLASSES,
        NUM_CLASSES,
        IMAGE_SIZE,
        BATCH_SIZE,
        RANDOM_SEED,
        TRAIN_DIR,
        VALID_DIR,
        TEST_DIR,
        compute_train_class_weights,
    )
    from model import build_cargo_cnn


# -----------------------------------------------------------------------------
# Configuration & Argument Parsing
# -----------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Cargo Vision CNN on 5-class classification dataset."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial Adam learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs (default: 15)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save trained model weights (default: models)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/training",
        help="Directory to save evaluation plots and logs (default: results/training)",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Data Pipeline Builder
# -----------------------------------------------------------------------------

def create_data_pipeline(
    batch_size: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads train, validation, and test datasets.
    Applies on-the-fly data augmentation strictly to the TRAIN split.
    """
    print("\n[Step 1/5] Loading and configuring TensorFlow dataset pipelines...")

    # Set global seeds
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # 1. Load raw datasets from directory structure
    train_raw_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=TARGET_CLASSES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        labels="inferred",
        label_mode="int",
        class_names=TARGET_CLASSES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=TARGET_CLASSES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    # 2. Build on-the-fly Data Augmentation block (Applied to TRAIN ONLY)
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=RANDOM_SEED),
            tf.keras.layers.RandomRotation(0.03, seed=RANDOM_SEED),  # +/- ~10.8 deg
            tf.keras.layers.RandomZoom(height_factor=(-0.08, 0.08), width_factor=(-0.08, 0.08), seed=RANDOM_SEED),
            tf.keras.layers.RandomTranslation(height_factor=(-0.06, 0.06), width_factor=(-0.06, 0.06), seed=RANDOM_SEED),
        ],
        name="train_augmentation_pipeline",
    )

    # Map augmentation to train dataset with parallel calls
    train_ds = train_raw_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    valid_ds = valid_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    print("  -> Data augmentation active for: TRAIN split")
    print("  -> Augmentation disabled for: VALIDATION and TEST splits")
    return train_ds, valid_ds, test_ds


# -----------------------------------------------------------------------------
# Diagnostic Visualizations & Reporting
# -----------------------------------------------------------------------------

def plot_training_curves(history: tf.keras.callbacks.History, save_path: str) -> None:
    """
    Plots and saves loss and accuracy learning curves across epochs.
    """
    hist = history.history
    epochs_range = range(1, len(hist["loss"]) + 1)
    best_epoch = int(np.argmin(hist["val_loss"])) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs_range, hist["loss"], label="Train Loss", color="#2563eb", linewidth=2)
    ax1.plot(epochs_range, hist["val_loss"], label="Val Loss", color="#dc2626", linewidth=2, linestyle="--")
    ax1.axvline(best_epoch, color="#16a34a", linestyle=":", label=f"Best Epoch ({best_epoch})")
    ax1.set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Sparse Categorical Crossentropy", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # Accuracy Curve
    ax2.plot(epochs_range, hist["accuracy"], label="Train Accuracy", color="#2563eb", linewidth=2)
    ax2.plot(epochs_range, hist["val_accuracy"], label="Val Accuracy", color="#dc2626", linewidth=2, linestyle="--")
    ax2.axvline(best_epoch, color="#16a34a", linestyle=":", label=f"Best Epoch ({best_epoch})")
    ax2.set_title("Training vs Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epochs", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    plt.suptitle("Cargo Vision CNN - Training History Curves", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved training curves to '{save_path}'.")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
) -> None:
    """
    Renders and saves a formatted confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    ax.set_title("Test Set Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Ground Truth Class", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved confusion matrix plot to '{save_path}'.")


# -----------------------------------------------------------------------------
# Main Training & Evaluation Flow
# -----------------------------------------------------------------------------

def main():
    args = parse_arguments()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 70)
    print("Cargo Vision CNN - Training & Evaluation Pipeline")
    print(f"Target Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.learning_rate}")
    print(f"Models Directory: {args.models_dir} | Results Directory: {args.results_dir}")
    print("=" * 70)

    # 1. Prepare Data Pipelines
    train_ds, valid_ds, test_ds = create_data_pipeline(batch_size=args.batch_size)

    # 2. Compute Class Weights on Train Split
    print("\n[Step 2/5] Computing balanced class weights from Train split...")
    class_weights = compute_train_class_weights()
    for idx, w in class_weights.items():
        print(f"  - Class {idx} ('{TARGET_CLASSES[idx]}'): Weight = {w:.4f}")

    # 3. Build Model from scripts/model.py
    print("\n[Step 3/5] Instantiating custom CNN architecture from scripts/model.py...")
    model = build_cargo_cnn(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=NUM_CLASSES,
        learning_rate=args.learning_rate,
    )

    # 4. Setup Callbacks
    best_model_path = os.path.join(args.models_dir, "cargo_cnn_best.keras")
    final_model_path = os.path.join(args.models_dir, "cargo_cnn_final.keras")
    history_csv_path = os.path.join(args.results_dir, "training_history.csv")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=history_csv_path,
            separator=",",
            append=False,
        ),
    ]

    # 5. Execute Training Loop
    print("\n[Step 4/5] Initiating model training...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    training_duration = time.time() - start_time
    print(f"\nTraining completed in {training_duration:.2f} seconds.")

    # Save final model state
    model.save(final_model_path)
    print(f"Saved final model to '{final_model_path}'.")

    # Generate Learning Curves
    curves_path = os.path.join(args.results_dir, "training_curves.png")
    plot_training_curves(history, curves_path)

    # 6. Comprehensive Evaluation on Held-Out Test Set
    print("\n[Step 5/5] Evaluating best model on held-out unseen test split...")
    
    # Load best checkpoint explicitly for evaluation
    best_model = tf.keras.models.load_model(best_model_path)

    # Collect ground-truth labels and model predictions
    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_pred_probs = best_model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    # Save Confusion Matrix Plot
    cm_path = os.path.join(args.results_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, TARGET_CLASSES, cm_path)

    # Classification Report
    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=TARGET_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    # Best Epoch Stats
    hist = history.history
    best_epoch_idx = int(np.argmin(hist["val_loss"]))
    best_epoch = best_epoch_idx + 1
    best_val_loss = float(hist["val_loss"][best_epoch_idx])
    best_val_acc = float(hist["val_accuracy"][best_epoch_idx])

    # Save Summary JSON
    summary_data = {
        "best_epoch": best_epoch,
        "total_epochs_trained": len(hist["loss"]),
        "training_duration_seconds": round(training_duration, 2),
        "best_val_loss": round(best_val_loss, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_macro_f1": round(float(macro_f1), 4),
        "test_weighted_f1": round(float(weighted_f1), 4),
        "class_weights": class_weights,
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    summary_json_path = os.path.join(args.results_dir, "training_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved machine-readable summary to '{summary_json_path}'.")

    # -------------------------------------------------------------------------
    # Final Console Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Cargo Vision CNN - Final Training & Evaluation Summary")
    print("=" * 70)
    print(f"Best Training Epoch:            {best_epoch} / {len(hist['loss'])}")
    print(f"Best Validation Loss:           {best_val_loss:.4f}")
    print(f"Best Validation Accuracy:       {best_val_acc * 100:.2f}%")
    print(f"Held-Out Test Accuracy:         {test_acc * 100:.2f}%")
    print(f"Held-Out Test Macro F1:         {macro_f1:.4f}")
    print(f"Held-Out Test Weighted F1:      {weighted_f1:.4f}")
    print("-" * 70)
    print(f"{'Class Name':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Support':<8}")
    print("-" * 70)
    for c in TARGET_CLASSES:
        m = cls_report.get(c, {})
        p = m.get("precision", 0.0)
        r = m.get("recall", 0.0)
        f = m.get("f1-score", 0.0)
        sup = int(m.get("support", 0))
        print(f"{c:<12} | {p:<12.4f} | {r:<12.4f} | {f:<12.4f} | {sup:<8}")
    print("=" * 70)


if __name__ == "__main__":
    main()
