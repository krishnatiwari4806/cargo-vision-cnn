"""
train_mobilenet_expanded.py - MobileNetV2 Transfer Learning on Expanded Dataset
================================================================================
Project: Cargo Vision CNN
Purpose: Evaluates MobileNetV2 transfer learning (frozen feature extractor)
         on the expanded 473-crop dataset (Train: 391, Valid: 49, Test: 33).
         Directly compares results with the original 401-crop benchmark.
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
from tensorflow.keras import layers, models, optimizers, losses, metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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


# -----------------------------------------------------------------------------
# Data Pipeline
# -----------------------------------------------------------------------------

def create_data_pipeline(
    batch_size: int = BATCH_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Constructs deterministic pipelines for validation/test and an augmented pipeline for train.
    """
    print("\n[Step 1/5] Loading expanded dataset splits...")
    tf.keras.utils.set_random_seed(RANDOM_SEED)

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

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=RANDOM_SEED),
            tf.keras.layers.RandomRotation(0.03, seed=RANDOM_SEED),
            tf.keras.layers.RandomZoom(height_factor=(-0.08, 0.08), width_factor=(-0.08, 0.08), seed=RANDOM_SEED),
            tf.keras.layers.RandomTranslation(height_factor=(-0.06, 0.06), width_factor=(-0.06, 0.06), seed=RANDOM_SEED),
        ],
        name="train_augmentation_pipeline",
    )

    train_ds = train_raw_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    valid_ds = valid_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, valid_ds, test_ds


# -----------------------------------------------------------------------------
# Architecture Definition
# -----------------------------------------------------------------------------

def build_mobilenetv2_classifier(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 5,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Constructs the standard transfer learning model with frozen MobileNetV2 backbone.
    """
    inputs = layers.Input(shape=input_shape, name="cargo_image_input")
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_rescaling")(inputs)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, name="dense_fc1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    x = layers.Dropout(0.30, name="dropout_dense")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="cargo_classification_output",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cargo_vision_mobilenetv2_expanded")

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    loss_fn = losses.SparseCategoricalCrossentropy()
    acc_metric = metrics.SparseCategoricalAccuracy(name="accuracy")

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[acc_metric])
    return model


# -----------------------------------------------------------------------------
# Plotting & Reporting
# -----------------------------------------------------------------------------

def plot_training_curves(history: tf.keras.callbacks.History, save_path: str) -> None:
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    epochs_range = range(1, len(loss) + 1)
    best_epoch = int(np.argmin(val_loss)) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs_range, loss, label="Training Loss", color="#2563eb", linewidth=2)
    ax1.plot(epochs_range, val_loss, label="Validation Loss", color="#dc2626", linewidth=2, linestyle="--")
    ax1.axvline(best_epoch, color="#16a34a", linestyle=":", label=f"Best Epoch ({best_epoch})")
    ax1.set_title("MobileNetV2 (Expanded): Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Crossentropy Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # Accuracy
    ax2.plot(epochs_range, acc, label="Training Accuracy", color="#2563eb", linewidth=2)
    ax2.plot(epochs_range, val_acc, label="Validation Accuracy", color="#dc2626", linewidth=2, linestyle="--")
    ax2.axvline(best_epoch, color="#16a34a", linestyle=":", label=f"Best Epoch ({best_epoch})")
    ax2.set_title("MobileNetV2 (Expanded): Training vs Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    plt.suptitle("Cargo Vision - MobileNetV2 (Expanded Dataset) Learning Curves", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved training curves to '{save_path}'.")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str) -> None:
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

    ax.set_title("MobileNetV2 (Expanded Dataset) Test Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Ground Truth Class", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved confusion matrix to '{save_path}'.")


# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------

def main():
    epochs = 30
    models_dir = "models"
    results_dir = "results/training"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    best_model_path = os.path.join(models_dir, "cargo_mobilenetv2_expanded_best.keras")
    history_csv_path = os.path.join(results_dir, "mobilenet_expanded_training_history.csv")
    curves_path = os.path.join(results_dir, "mobilenet_expanded_training_curves.png")
    cm_path = os.path.join(results_dir, "mobilenet_expanded_confusion_matrix.png")
    summary_json_path = os.path.join(results_dir, "mobilenet_expanded_training_summary.json")

    print("=" * 78)
    print("Cargo Vision CNN - MobileNetV2 (Expanded 473-Crop Dataset) Training")
    print(f"Epochs: {epochs} | Batch Size: {BATCH_SIZE} | Checkpoint: {best_model_path}")
    print("=" * 78)

    # 1. Pipeline & Class Weights
    train_ds, valid_ds, test_ds = create_data_pipeline(batch_size=BATCH_SIZE)
    class_weights = compute_train_class_weights()
    print("\n[Step 2/5] Balanced Class Weights calculated for 391 training samples:")
    for idx, name in enumerate(TARGET_CLASSES):
        print(f"  Class {idx} ({name:<10}): {class_weights[idx]:.4f}")

    # 2. Build Model
    print("\n[Step 3/5] Building MobileNetV2 architecture with frozen ImageNet base...")
    model = build_mobilenetv2_classifier(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=NUM_CLASSES,
        learning_rate=0.001,
    )
    model.summary(line_length=78)

    # 3. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
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
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=history_csv_path,
            separator=",",
            append=False,
        ),
    ]

    # 4. Fit Model
    print("\n[Step 4/5] Executing training loop on expanded training split (N=391)...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Plot Curves
    plot_training_curves(history, curves_path)

    # 5. Evaluate on Expanded Test Split (N=33)
    print("\n[Step 5/5] Evaluating best checkpoint on expanded held-out test split (N=33)...")
    best_model = tf.keras.models.load_model(best_model_path)

    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_pred_probs = best_model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    plot_confusion_matrix(cm, TARGET_CLASSES, cm_path)

    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=TARGET_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    best_epoch_idx = int(np.argmin(history.history["val_loss"]))
    best_val_loss = float(history.history["val_loss"][best_epoch_idx])
    best_val_acc = float(history.history["val_accuracy"][best_epoch_idx])

    summary_data = {
        "model_architecture": "MobileNetV2 (Frozen Feature Extractor, Expanded Dataset)",
        "dataset_split_counts": {
            "train": 391,
            "valid": 49,
            "test": 33,
            "total": 473,
        },
        "epochs_trained": len(history.history["loss"]),
        "best_epoch": best_epoch_idx + 1,
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

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved machine-readable summary to '{summary_json_path}'.")

    # Benchmark Comparison
    old_acc, old_mf1, old_wf1 = 0.6364, 0.5562, 0.6446

    print("\n" + "=" * 78)
    print("Cargo Vision - MobileNetV2 (Expanded Dataset) Evaluation Results")
    print("=" * 78)
    print(f"Best Validation Loss:           {best_val_loss:.4f} (Epoch {best_epoch_idx + 1})")
    print(f"Best Validation Accuracy:       {best_val_acc * 100:.2f}%")
    print(f"Held-Out Test Accuracy:         {test_acc * 100:.2f}% ({int(test_acc * len(y_true))}/{len(y_true)})")
    print(f"Held-Out Test Macro F1:         {macro_f1:.4f}")
    print(f"Held-Out Test Weighted F1:      {weighted_f1:.4f}")
    print("-" * 78)
    print(f"{'Class Name':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Support':<8}")
    print("-" * 78)
    for c in TARGET_CLASSES:
        m = cls_report.get(c, {})
        p = m.get("precision", 0.0)
        r = m.get("recall", 0.0)
        f = m.get("f1-score", 0.0)
        sup = int(m.get("support", 0))
        print(f"{c:<12} | {p:<12.4f} | {r:<12.4f} | {f:<12.4f} | {sup:<8}")
    print("=" * 78)

    print("\n" + "=" * 80)
    print("DATASET EXPANSION BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<24} | {'Old MobileNetV2 (401 Crops)':<26} | {'New MobileNetV2 (473 Crops)':<26}")
    print("-" * 80)
    print(f"{'Dataset Size (Tr/Va/Te)':<24} | {'341 / 38 / 22':<26} | {'391 / 49 / 33':<26}")
    print(f"{'Test Accuracy':<24} | {old_acc*100:>23.2f}% | {test_acc*100:>23.2f}%")
    print(f"{'Macro F1':<24} | {old_mf1:>24.4f} | {macro_f1:>24.4f}")
    print(f"{'Weighted F1':<24} | {old_wf1:>24.4f} | {weighted_f1:>24.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
