"""
train_mobilenet.py - Transfer Learning Pipeline (MobileNetV2) for Cargo Vision
==============================================================================
Project: Cargo Vision CNN
Purpose: Fine-tunes a pre-trained MobileNetV2 backbone (ImageNet weights)
         on the 5-class cargo dataset to benchmark against the from-scratch CNN.

Workflow:
  1. Loads dataset splits (data/classification/{train, valid, test}).
  2. Applies conservative on-the-fly data augmentation strictly to TRAIN split.
  3. Computes balanced class weights on Train split to compensate for imbalance.
  4. Builds MobileNetV2 model with frozen ImageNet feature extractor and compact classifier head.
  5. Trains with EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, and CSVLogger.
  6. Evaluates the best model on the same held-out test split (N=22).
  7. Generates training curves, confusion matrix, and classification report.
  8. Saves best model to models/cargo_mobilenetv2_best.keras and logs to results/training/.
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
# MobileNetV2 Model Builder
# -----------------------------------------------------------------------------

def build_mobilenetv2_classifier(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 5,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Constructs a transfer-learning model using a frozen MobileNetV2 ImageNet backbone.
    """
    inputs = layers.Input(shape=input_shape, name="cargo_image_input")

    # MobileNetV2 preprocessing: scales [0, 255] to [-1.0, 1.0]
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_rescaling")(inputs)

    # Pretrained MobileNetV2 base (frozen)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(x, training=False)

    # Classification Head
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

    model = models.Model(inputs=inputs, outputs=outputs, name="cargo_vision_mobilenetv2")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


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
    print("\n[Step 1/5] Loading and configuring dataset pipelines...")

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

    # Train-only conservative augmentation
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

    print("  -> Data augmentation active for: TRAIN split")
    print("  -> Augmentation disabled for: VALIDATION and TEST splits")
    return train_ds, valid_ds, test_ds


# -----------------------------------------------------------------------------
# Diagnostic Visualizations
# -----------------------------------------------------------------------------

def plot_training_curves(history: tf.keras.callbacks.History, save_path: str) -> None:
    hist = history.history
    epochs_range = range(1, len(hist["loss"]) + 1)
    best_epoch = int(np.argmin(hist["val_loss"])) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs_range, hist["loss"], label="Train Loss", color="#2563eb", linewidth=2)
    ax1.plot(epochs_range, hist["val_loss"], label="Val Loss", color="#dc2626", linewidth=2, linestyle="--")
    ax1.axvline(best_epoch, color="#16a34a", linestyle=":", label=f"Best Epoch ({best_epoch})")
    ax1.set_title("MobileNetV2: Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Sparse Categorical Crossentropy", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # Accuracy
    ax2.plot(epochs_range, hist["accuracy"], label="Train Accuracy", color="#2563eb", linewidth=2)
    ax2.plot(epochs_range, hist["val_accuracy"], label="Val Accuracy", color="#dc2626", linewidth=2, linestyle="--")
    ax2.axvline(best_epoch, color="#16a34a", linestyle=":", label=f"Best Epoch ({best_epoch})")
    ax2.set_title("MobileNetV2: Training vs Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epochs", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    plt.suptitle("Cargo Vision - MobileNetV2 Transfer Learning History", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved training curves to '{save_path}'.")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
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

    ax.set_title("MobileNetV2 Test Set Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Ground Truth Class", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved confusion matrix plot to '{save_path}'.")


# -----------------------------------------------------------------------------
# Main Training & Evaluation
# -----------------------------------------------------------------------------

def main():
    epochs = 40
    batch_size = BATCH_SIZE
    learning_rate = 0.001
    patience = 15

    models_dir = "models"
    results_dir = "results/training"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 75)
    print("Cargo Vision CNN - Transfer Learning Pipeline (MobileNetV2 Backbone)")
    print(f"Max Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    print("=" * 75)

    # 1. Prepare Data
    train_ds, valid_ds, test_ds = create_data_pipeline(batch_size=batch_size)

    # 2. Balanced Class Weights
    print("\n[Step 2/5] Computing balanced class weights from Train split...")
    class_weights = compute_train_class_weights()
    for idx, w in class_weights.items():
        print(f"  - Class {idx} ('{TARGET_CLASSES[idx]}'): Weight = {w:.4f}")

    # 3. Build Model
    print("\n[Step 3/5] Instantiating MobileNetV2 transfer-learning model...")
    model = build_mobilenetv2_classifier(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=NUM_CLASSES,
        learning_rate=learning_rate,
    )
    model.summary()

    # 4. Setup Callbacks
    best_model_path = os.path.join(models_dir, "cargo_mobilenetv2_best.keras")
    history_csv_path = os.path.join(results_dir, "mobilenet_training_history.csv")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
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

    # 5. Execute Training
    print("\n[Step 4/5] Initiating MobileNetV2 transfer-learning training...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    training_duration = time.time() - start_time
    print(f"\nTraining completed in {training_duration:.2f} seconds.")

    # Generate Learning Curves
    curves_path = os.path.join(results_dir, "mobilenet_training_curves.png")
    plot_training_curves(history, curves_path)

    # 6. Evaluation on Held-Out Test Split
    print("\n[Step 5/5] Evaluating best MobileNetV2 model on held-out test split...")
    best_model = tf.keras.models.load_model(best_model_path)

    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_pred_probs = best_model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    # Save Confusion Matrix Plot
    cm_path = os.path.join(results_dir, "mobilenet_confusion_matrix.png")
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
        "model_architecture": "MobileNetV2 (ImageNet Pretrained Feature Extractor)",
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

    summary_json_path = os.path.join(results_dir, "mobilenet_training_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved machine-readable summary to '{summary_json_path}'.")

    # -------------------------------------------------------------------------
    # Baseline vs MobileNetV2 Comparison Table
    # -------------------------------------------------------------------------
    # Baseline values
    baseline_acc = 0.3182
    baseline_macro_f1 = 0.1000
    baseline_weighted_f1 = 0.1591

    print("\n" + "=" * 75)
    print("Cargo Vision - MobileNetV2 Evaluation Results")
    print("=" * 75)
    print(f"Best Training Epoch:            {best_epoch} / {len(hist['loss'])}")
    print(f"Best Validation Loss:           {best_val_loss:.4f}")
    print(f"Best Validation Accuracy:       {best_val_acc * 100:.2f}%")
    print(f"Held-Out Test Accuracy:         {test_acc * 100:.2f}%")
    print(f"Held-Out Test Macro F1:         {macro_f1:.4f}")
    print(f"Held-Out Test Weighted F1:      {weighted_f1:.4f}")
    print("-" * 75)
    print(f"{'Class Name':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Support':<8}")
    print("-" * 75)
    for c in TARGET_CLASSES:
        m = cls_report.get(c, {})
        p = m.get("precision", 0.0)
        r = m.get("recall", 0.0)
        f = m.get("f1-score", 0.0)
        sup = int(m.get("support", 0))
        print(f"{c:<12} | {p:<12.4f} | {r:<12.4f} | {f:<12.4f} | {sup:<8}")
    print("=" * 75)

    print("\n" + "=" * 75)
    print("ARCHITECTURAL BENCHMARK COMPARISON")
    print("=" * 75)
    print(f"{'Metric':<25} | {'From-Scratch CNN':<20} | {'MobileNetV2 (Transfer)':<20} | {'Delta':<10}")
    print("-" * 75)
    delta_acc = (test_acc - baseline_acc) * 100
    delta_mf1 = macro_f1 - baseline_macro_f1
    delta_wf1 = weighted_f1 - baseline_weighted_f1
    print(f"{'Test Accuracy':<25} | {baseline_acc * 100:>18.2f}% | {test_acc * 100:>18.2f}% | {delta_acc:>+9.2f}%")
    print(f"{'Macro F1':<25} | {baseline_macro_f1:>19.4f} | {macro_f1:>19.4f} | {delta_mf1:>+9.4f}")
    print(f"{'Weighted F1':<25} | {baseline_weighted_f1:>19.4f} | {weighted_f1:>19.4f} | {delta_wf1:>+9.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
