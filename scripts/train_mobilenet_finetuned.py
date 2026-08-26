"""
train_mobilenet_finetuned.py - Controlled Fine-Tuning of MobileNetV2
====================================================================
Project: Cargo Vision CNN
Purpose: Executes a two-phase transfer learning & fine-tuning strategy:
         Phase 1: Warmup classification head with frozen MobileNetV2 backbone.
         Phase 2: Unfreeze top semantic layers (block_13..16 & Conv_1) with low LR (1e-4)
                  while keeping BatchNorm layers frozen to stabilize batch statistics.
         Evaluates on the exact same held-out test split (N=22).
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
    batch_size: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads train, validation, and test datasets.
    Applies on-the-fly conservative data augmentation strictly to TRAIN split.
    """
    print("\n[Step 1/5] Configuring dataset pipelines...")
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
# Model Construction
# -----------------------------------------------------------------------------

def build_finetunable_mobilenet(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 5,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Builds the base MobileNetV2 and complete classifier model.
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

    model = models.Model(inputs=inputs, outputs=outputs, name="cargo_vision_mobilenetv2_finetuned")
    return model, base_model


# -----------------------------------------------------------------------------
# Plotting & Reporting
# -----------------------------------------------------------------------------

def plot_finetuning_curves(
    history_phase1: Dict[str, List[float]],
    history_phase2: Dict[str, List[float]],
    save_path: str,
) -> None:
    """
    Plots combined learning curves spanning Phase 1 (Warmup) and Phase 2 (Fine-tuning).
    """
    loss = history_phase1["loss"] + history_phase2["loss"]
    val_loss = history_phase1["val_loss"] + history_phase2["val_loss"]
    acc = history_phase1["accuracy"] + history_phase2["accuracy"]
    val_acc = history_phase1["val_accuracy"] + history_phase2["val_accuracy"]

    epochs_range = range(1, len(loss) + 1)
    phase1_len = len(history_phase1["loss"])
    best_epoch = int(np.argmin(val_loss)) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs_range, loss, label="Train Loss", color="#2563eb", linewidth=2)
    ax1.plot(epochs_range, val_loss, label="Val Loss", color="#dc2626", linewidth=2, linestyle="--")
    ax1.axvline(phase1_len, color="#6b7280", linestyle="-.", label="Fine-Tuning Start")
    ax1.axvline(best_epoch, color="#16a34a", linestyle=":", label=f"Best Epoch ({best_epoch})")
    ax1.set_title("Fine-Tuning: Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Total Epochs", fontsize=11)
    ax1.set_ylabel("Sparse Categorical Crossentropy", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # Accuracy
    ax2.plot(epochs_range, acc, label="Train Accuracy", color="#2563eb", linewidth=2)
    ax2.plot(epochs_range, val_acc, label="Val Accuracy", color="#dc2626", linewidth=2, linestyle="--")
    ax2.axvline(phase1_len, color="#6b7280", linestyle="-.", label="Fine-Tuning Start")
    ax2.axvline(best_epoch, color="#16a34a", linestyle=":", label=f"Best Epoch ({best_epoch})")
    ax2.set_title("Fine-Tuning: Training vs Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Total Epochs", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    plt.suptitle("Cargo Vision - MobileNetV2 Controlled Fine-Tuning Curves", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved combined fine-tuning curves to '{save_path}'.")


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

    ax.set_title("Fine-Tuned MobileNetV2 Test Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Ground Truth Class", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved confusion matrix plot to '{save_path}'.")


# -----------------------------------------------------------------------------
# Main Fine-Tuning Execution
# -----------------------------------------------------------------------------

def main():
    warmup_epochs = 15
    finetune_epochs = 35
    fine_tune_at = 115  # Unfreeze block_13 through block_16 & Conv_1 (top ~39 layers)

    models_dir = "models"
    results_dir = "results/training"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    finetuned_best_model_path = os.path.join(models_dir, "cargo_mobilenetv2_finetuned_best.keras")
    history_csv_path = os.path.join(results_dir, "mobilenet_finetuned_training_history.csv")

    print("=" * 78)
    print("Cargo Vision CNN - MobileNetV2 Controlled Fine-Tuning Pipeline")
    print(f"Warmup Epochs: {warmup_epochs} | Fine-Tuning Epochs: {finetune_epochs}")
    print(f"Fine-Tuning Cutoff Layer: {fine_tune_at} / 154 (Top 39 layers unfreezed)")
    print("=" * 78)

    # 1. Pipeline & Class Weights
    train_ds, valid_ds, test_ds = create_data_pipeline(batch_size=BATCH_SIZE)
    class_weights = compute_train_class_weights()

    # 2. Build Model
    model, base_model = build_finetunable_mobilenet(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=NUM_CLASSES,
    )

    # =========================================================================
    # PHASE 1: Warmup Classification Head (Frozen Base)
    # =========================================================================
    print("\n[Phase 1/2] Training classification head with frozen MobileNetV2 base...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    h1 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=warmup_epochs,
        class_weight=class_weights,
        verbose=1,
    )

    # =========================================================================
    # PHASE 2: Controlled Fine-Tuning of Top Semantic Layers
    # =========================================================================
    print(f"\n[Phase 2/2] Unfreezing MobileNetV2 layers from index {fine_tune_at} onwards...")
    base_model.trainable = True

    # Freeze earlier layers & keep all BatchNormalization layers frozen
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    for layer in base_model.layers[fine_tune_at:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False  # Freeze BN statistics for stability on small dataset
        else:
            layer.trainable = True

    trainable_count = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"  -> Total Trainable Parameters in Fine-Tuning: {trainable_count:,}")

    # Recompile with gentle learning rate
    fine_tune_lr = 1e-4
    model.compile(
        optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    callbacks_phase2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=finetuned_best_model_path,
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

    h2 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=finetune_epochs,
        class_weight=class_weights,
        callbacks=callbacks_phase2,
        verbose=1,
    )

    # Plot Combined Curves
    curves_path = os.path.join(results_dir, "mobilenet_finetuned_training_curves.png")
    plot_finetuning_curves(h1.history, h2.history, curves_path)

    # =========================================================================
    # EVALUATION ON UNTOUCHED HELD-OUT TEST SPLIT (N=22)
    # =========================================================================
    print("\n[Evaluation] Evaluating best fine-tuned checkpoint on held-out test split...")
    best_model = tf.keras.models.load_model(finetuned_best_model_path)

    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_pred_probs = best_model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    # Save Confusion Matrix Plot
    cm_path = os.path.join(results_dir, "mobilenet_finetuned_confusion_matrix.png")
    plot_confusion_matrix(cm, TARGET_CLASSES, cm_path)

    # Classification Report
    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=TARGET_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    # Best Epoch Stats in Phase 2
    best_epoch_idx = int(np.argmin(h2.history["val_loss"]))
    best_val_loss = float(h2.history["val_loss"][best_epoch_idx])
    best_val_acc = float(h2.history["val_accuracy"][best_epoch_idx])

    # Save Summary JSON
    summary_data = {
        "model_architecture": "MobileNetV2 (Fine-Tuned Top 39 Layers, BN Frozen)",
        "phase1_warmup_epochs": warmup_epochs,
        "phase2_finetune_epochs_trained": len(h2.history["loss"]),
        "finetune_learning_rate": fine_tune_lr,
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

    summary_json_path = os.path.join(results_dir, "mobilenet_finetuned_training_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved machine-readable summary to '{summary_json_path}'.")

    # -------------------------------------------------------------------------
    # Comprehensive 3-Way Benchmark Comparison
    # -------------------------------------------------------------------------
    scratch_acc, scratch_mf1, scratch_wf1 = 0.3182, 0.1000, 0.1591
    frozen_acc, frozen_mf1, frozen_wf1 = 0.6364, 0.5562, 0.6446

    print("\n" + "=" * 78)
    print("Cargo Vision - Fine-Tuned MobileNetV2 Evaluation Results")
    print("=" * 78)
    print(f"Best Validation Loss:           {best_val_loss:.4f}")
    print(f"Best Validation Accuracy:       {best_val_acc * 100:.2f}%")
    print(f"Held-Out Test Accuracy:         {test_acc * 100:.2f}%")
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
    print("3-WAY ARCHITECTURAL BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<20} | {'From-Scratch CNN':<16} | {'Frozen MobileNet':<16} | {'Fine-Tuned MobileNet':<20}")
    print("-" * 80)
    print(f"{'Test Accuracy':<20} | {scratch_acc*100:>15.2f}% | {frozen_acc*100:>15.2f}% | {test_acc*100:>19.2f}%")
    print(f"{'Macro F1':<20} | {scratch_mf1:>16.4f} | {frozen_mf1:>16.4f} | {macro_f1:>20.4f}")
    print(f"{'Weighted F1':<20} | {scratch_wf1:>16.4f} | {frozen_wf1:>16.4f} | {weighted_f1:>20.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
