"""
predict_cargo.py - Cargo Prediction & Vehicle Recommendation Engine
===================================================================
Project: Cargo Vision CNN
Purpose: Takes an input cargo image, classifies it using the trained CNN
         (MobileNetV2 or from-scratch CNN), displays class probability breakdown,
         and provides prototype rule-based vehicle suitability recommendations.

Important Engineering Note:
  Vehicle recommendations are heuristic rule-based estimations based on classified
  cargo category. Single 2D RGB camera images cannot determine exact physical weight (kg)
  or exact 3D dimensions (cm).
"""

import os
import sys
import argparse
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from scripts.dataset_config import (
        TARGET_CLASSES,
        INDEX_TO_CLASS,
        IMAGE_SIZE,
    )
except ModuleNotFoundError:
    from dataset_config import (
        TARGET_CLASSES,
        INDEX_TO_CLASS,
        IMAGE_SIZE,
    )


# -----------------------------------------------------------------------------
# Vehicle Recommendation Knowledge Base (Prototype Business Heuristics)
# -----------------------------------------------------------------------------

VEHICLE_CATALOG: Dict[str, Dict[str, Any]] = {
    "box": {
        "primary_vehicle": "Mini Cargo Tempo / 3-Wheeler (e.g., Bajaj Maxima / Tata Ace Zip)",
        "capacity_class": "Light Parcel / Box (Payload: Up to 500 kg)",
        "alternative_vehicle": "Standard Pickup or 2-Wheeler (if single small parcel)",
        "handling_notes": "Stackable, requires secure tie-downs to prevent shifting during transit.",
    },
    "suitcase": {
        "primary_vehicle": "Cargo Auto / Hatchback / Sedan Luggage Carrier",
        "capacity_class": "Personal Cargo / Luggage (Payload: Up to 150 kg)",
        "alternative_vehicle": "Mini Van (e.g., Maruti Eeco Cargo)",
        "handling_notes": "Fragile handles and wheels; keep dry and avoid heavy top-loading.",
    },
    "chair": {
        "primary_vehicle": "Small Commercial Pickup (e.g., Tata Ace / Mahindra Bolero Maxi Truck)",
        "capacity_class": "Furniture - Medium (Payload: Up to 750 kg)",
        "alternative_vehicle": "Flatbed Tempo",
        "handling_notes": "Protruding legs require corner padding and blanket wrapping.",
    },
    "table": {
        "primary_vehicle": "Light Commercial Truck / Flatbed (e.g., Tata 407 / Ashok Leyland Dost)",
        "capacity_class": "Furniture - Heavy / Bulky (Payload: Up to 1.5 Tons)",
        "alternative_vehicle": "14-ft Container Truck",
        "handling_notes": "Flat surface requires protective felt padding; load flat or dismantle legs if possible.",
    },
    "couch": {
        "primary_vehicle": "Large Box Truck / Moving Van (e.g., Eicher 14ft - 19ft Container Truck)",
        "capacity_class": "Heavy Bulky Furniture (Payload: Up to 3.0 Tons)",
        "alternative_vehicle": "Enclosed 20ft Truck for multi-item relocation",
        "handling_notes": "High spatial volume; weatherproof enclosed carriage strictly recommended.",
    },
}


# -----------------------------------------------------------------------------
# Image Preprocessing & Inference
# -----------------------------------------------------------------------------

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Opens, validates, and resizes an input image to (1, 128, 128, 3).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        img_resized = img_rgb.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32)

    # Add batch dimension: (1, 128, 128, 3)
    return np.expand_dims(img_array, axis=0)


def predict_cargo(
    model: tf.keras.Model,
    image_tensor: np.ndarray,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Executes inference and returns:
      - predicted class name
      - confidence score
      - full probability distribution dictionary
    """
    probs = model.predict(image_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = INDEX_TO_CLASS[pred_idx]
    confidence = float(probs[pred_idx])

    prob_distribution = {
        TARGET_CLASSES[i]: float(probs[i]) for i in range(len(TARGET_CLASSES))
    }
    return pred_class, confidence, prob_distribution


# -----------------------------------------------------------------------------
# Recommendation Formatter
# -----------------------------------------------------------------------------

def display_prediction_report(
    image_path: str,
    model_name: str,
    pred_class: str,
    confidence: float,
    prob_distribution: Dict[str, float],
) -> None:
    """
    Renders a formatted CLI report with cargo classification and vehicle recommendation.
    """
    print("\n" + "=" * 78)
    print("CARGO VISION: INFERENCE & VEHICLE RECOMMENDATION REPORT")
    print("=" * 78)
    print(f"Input Image:      {image_path}")
    print(f"Inference Model:  {model_name}")
    print(f"Predicted Cargo:  {pred_class.upper()} (Confidence: {confidence * 100:.2f}%)")
    print("-" * 78)

    print("\n[Softmax Probability Distribution Across All Classes]")
    print(f"{'Class Name':<12} | {'Probability':<14} | {'Visual Bar'}")
    print("-" * 78)
    for cname in TARGET_CLASSES:
        p = prob_distribution[cname]
        bar_len = int(round(p * 35))
        bar = "#" * bar_len
        marker = " <== PREDICTED" if cname == pred_class else ""
        print(f"{cname:<12} | {p * 100:>6.2f}%        | {bar}{marker}")

    # Vehicle Recommendation Section
    rec = VEHICLE_CATALOG.get(pred_class, {})
    print("\n" + "=" * 78)
    print("LOGISTICS & VEHICLE SUITABILITY RECOMMENDATION")
    print("=" * 78)
    print(f"Primary Recommended Vehicle: {rec.get('primary_vehicle', 'Standard Cargo Truck')}")
    print(f"Cargo Capacity Class:        {rec.get('capacity_class', 'General Logistics')}")
    print(f"Alternative Option:          {rec.get('alternative_vehicle', 'N/A')}")
    print(f"Special Handling Notes:      {rec.get('handling_notes', 'Standard handling')}")
    print("-" * 78)
    print("DISCLAIMER: Prototype vehicle recommendation is based on category heuristics.")
    print("            Single 2D camera images do not compute exact kg weight or 3D volume.")
    print("=" * 78 + "\n")


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cargo Vision: Single-Image Cargo Classifier & Vehicle Recommender"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=False,
        default=None,
        help="Path to the input cargo image file (defaults to first available test sample).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["mobilenet", "scratch"],
        default="mobilenet",
        help="Model architecture to use: 'mobilenet' (recommended, 63.64% test acc) or 'scratch' (custom baseline).",
    )
    args = parser.parse_args()

    # Determine test image if none provided
    if not args.image:
        import glob
        test_samples = glob.glob(os.path.join(DATASET_ROOT if 'DATASET_ROOT' in globals() else "data/classification", "test", "*", "*.jpg"))
        if test_samples:
            args.image = test_samples[0]
            print(f"[Info] No input image specified. Using default test sample: {args.image}")
        else:
            print("[Error] No test images found in data/classification/test/. Please specify --image <path>.")
            sys.exit(1)


    # Determine model checkpoint path
    if args.model_type == "mobilenet":
        model_path = os.path.join("models", "cargo_mobilenetv2_best.keras")
        model_display = "MobileNetV2 Transfer Learning (Pretrained ImageNet)"
    else:
        model_path = os.path.join("models", "cargo_cnn_best.keras")
        model_display = "Custom From-Scratch CNN Baseline"

    if not os.path.exists(model_path):
        print(f"[Error] Model checkpoint not found at: {model_path}")
        print("Please train the model first using scripts/train_mobilenet.py or scripts/train.py.")
        sys.exit(1)

    print(f"Loading trained model from '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    # Preprocess image
    image_tensor = preprocess_image(args.image)

    # Run inference
    pred_class, confidence, prob_distribution = predict_cargo(model, image_tensor)

    # Display report
    display_prediction_report(
        image_path=args.image,
        model_name=model_display,
        pred_class=pred_class,
        confidence=confidence,
        prob_distribution=prob_distribution,
    )


if __name__ == "__main__":
    main()
