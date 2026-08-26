"""
model.py - Custom From-Scratch CNN Architecture for Cargo Classification
========================================================================
Project: Cargo Vision CNN
Purpose: Defines a lightweight, modular Convolutional Neural Network (CNN)
         tailored for 5-class cargo image classification on compact datasets.

Architecture Design:
  1. Input Rescaling: Normalizes pixel intensities from [0, 255] to [0.0, 1.0].
  2. Feature Extraction: 4 Conv2D + BatchNorm + ReLU blocks with progressive channel depth (32 -> 64 -> 128 -> 256).
  3. Spatial Reduction: MaxPooling2D (2x2) between blocks reduces dimensions (128 -> 64 -> 32 -> 16 -> 8).
  4. Regularization: BatchNormalization on every conv layer + progressive Dropout (0.2 -> 0.4) to combat overfitting.
  5. Parameter-Efficient Head: GlobalAveragePooling2D instead of a massive Flatten layer to minimize parameter count.
  6. Dense Classifier & Softmax: 128-unit dense layer followed by 5-unit Softmax probability distribution.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
from typing import Tuple

# -----------------------------------------------------------------------------
# Default Architecture Hyperparameters
# -----------------------------------------------------------------------------

DEFAULT_INPUT_SHAPE: Tuple[int, int, int] = (128, 128, 3)
DEFAULT_NUM_CLASSES: int = 5
DEFAULT_LEARNING_RATE: float = 0.001


def build_cargo_cnn(
    input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tf.keras.Model:
    """
    Constructs and compiles the custom from-scratch Cargo Vision CNN model.
    
    Args:
        input_shape: 3D tuple (height, width, channels), default (128, 128, 3).
        num_classes: Number of target classification categories, default 5.
        learning_rate: Initial learning rate for the Adam optimizer, default 0.001.
        
    Returns:
        Compiled tf.keras.Model instance.
    """
    inputs = layers.Input(shape=input_shape, name="cargo_image_input")

    # Layer 0: Normalization / Rescaling [0, 255] -> [0.0, 1.0]
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # -------------------------------------------------------------------------
    # Block 1: Low-level spatial features (edges, corners, lines)
    # Tensor Shape: (128, 128, 3) -> (128, 128, 32) -> (64, 64, 32)
    # -------------------------------------------------------------------------
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_1")(x)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)

    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.20, name="dropout1")(x)

    # -------------------------------------------------------------------------
    # Block 2: Intermediate textures and geometric patterns
    # Tensor Shape: (64, 64, 32) -> (64, 64, 64) -> (32, 32, 64)
    # -------------------------------------------------------------------------
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="dropout2")(x)

    # -------------------------------------------------------------------------
    # Block 3: Complex structural object parts (legs, cushions, panels, handles)
    # Tensor Shape: (32, 32, 64) -> (32, 32, 128) -> (16, 16, 128)
    # -------------------------------------------------------------------------
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.30, name="dropout3")(x)

    # -------------------------------------------------------------------------
    # Block 4: High-level semantic cargo representations
    # Tensor Shape: (16, 16, 128) -> (16, 16, 256) -> (8, 8, 256)
    # -------------------------------------------------------------------------
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="relu4_1")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    x = layers.Dropout(0.35, name="dropout4")(x)

    # -------------------------------------------------------------------------
    # Classification Head: GlobalAveragePooling2D + Dense + Softmax
    # GlobalAveragePooling reduces (8, 8, 256) to (256,) feature vector
    # -------------------------------------------------------------------------
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = layers.Dense(128, name="dense_fc1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    x = layers.Dropout(0.40, name="dropout_dense")(x)

    # Final Softmax Output across 5 cargo classes
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="cargo_classification_output",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cargo_vision_cnn_scratch")

    # Compile with Adam optimizer, Sparse Categorical Crossentropy, and Accuracy
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return model


# -----------------------------------------------------------------------------
# Verification & Self-Test
# -----------------------------------------------------------------------------

def verify_model_architecture() -> None:
    """
    Builds the model, executes a forward pass on a dummy tensor,
    and validates tensor shapes and parameter statistics.
    """
    print("=" * 70)
    print("Cargo Vision CNN - Architecture Definition & Verification")
    print("=" * 70)

    # 1. Build model
    model = build_cargo_cnn(
        input_shape=DEFAULT_INPUT_SHAPE,
        num_classes=DEFAULT_NUM_CLASSES,
        learning_rate=DEFAULT_LEARNING_RATE,
    )

    # 2. Print summary
    print("\n[Model Architecture Summary]")
    model.summary()

    # 3. Parameter counts
    total_params = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable_params = total_params - trainable_params

    print("\n" + "=" * 70)
    print(f"{'Parameter Metric':<30} | {'Count':<20}")
    print("-" * 70)
    print(f"{'Total Parameters':<30} | {total_params:,}")
    print(f"{'Trainable Parameters':<30} | {trainable_params:,}")
    print(f"{'Non-Trainable (BatchNorm)':<30} | {non_trainable_params:,}")
    print("=" * 70)

    # 4. Dummy forward-pass shape and output validity test
    batch_size = 4
    dummy_input = tf.random.uniform(
        shape=(batch_size, DEFAULT_INPUT_SHAPE[0], DEFAULT_INPUT_SHAPE[1], DEFAULT_INPUT_SHAPE[2]),
        minval=0.0,
        maxval=255.0,
        dtype=tf.float32,
    )

    print(f"\n[Forward Pass Test] Testing with dummy input tensor of shape: {dummy_input.shape}...")
    dummy_output = model(dummy_input, training=False)
    print(f"  -> Output tensor shape: {dummy_output.shape} (Expected: ({batch_size}, {DEFAULT_NUM_CLASSES}))")
    assert dummy_output.shape == (batch_size, DEFAULT_NUM_CLASSES), f"Shape mismatch: {dummy_output.shape}"

    # Verify softmax probability sum
    prob_sums = tf.reduce_sum(dummy_output, axis=1).numpy()
    print(f"  -> Softmax probability sums per batch sample: {prob_sums} (Expected: ~1.0 each)")
    assert tf.reduce_all(tf.abs(prob_sums - 1.0) < 1e-4), "Softmax probabilities do not sum to 1.0!"

    print("\n[PASSED] CNN model architecture verified successfully with zero errors!")


if __name__ == "__main__":
    verify_model_architecture()
