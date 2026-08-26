# Cargo Vision CNN: Model Handoff & Integration Specification

This document provides complete technical handoff instructions, input/output schemas, and drop-in integration code for backend and application engineers integrating the **Cargo Vision MobileNetV2** model.

---

## 1. Model Overview & Architecture

- **Model Name:** `cargo_vision_mobilenetv2`
- **Architecture Type:** Transfer Learning with Pre-trained ImageNet Backbone
- **Base Backbone:** `tf.keras.applications.MobileNetV2` (weights frozen)
- **Classification Head:**
  - `GlobalAveragePooling2D()`
  - `Dense(128, activation="relu")` + `BatchNormalization()` + `Dropout(0.30)`
  - `Dense(5, activation="softmax", name="cargo_classification_output")`
- **Total Parameters:** `2,423,749` ($\approx 9.24\text{ MB}$)
- **Framework Version:** TensorFlow 2.16+ / Keras 3.x

---

## 2. Model File Artifact

- **Model Path:** [`models/cargo_mobilenetv2_best.keras`](file:///c:/Users/ACER/OneDrive/Documents/cargo-vision-cnn/cargo-vision-cnn/models/cargo_mobilenetv2_best.keras)
- **Format:** Native Keras v3 Archive format (`.keras`), containing architecture, weights, and compilation state.

---

## 3. Expected Input Specification

| Attribute | Specification | Notes |
| :--- | :--- | :--- |
| **Spatial Dimensions** | **$128 \times 128\text{ pixels}$** | Resized with preserved aspect ratio or high-quality Lanczos interpolation. |
| **Color Channels** | **$3\text{ (RGB)}$** | Ensure images are converted to 3-channel RGB (strip alpha channel if PNG). |
| **Batch Tensor Shape** | `(batch_size, 128, 128, 3)` | For single image: `(1, 128, 128, 3)` with `dtype=float32` or `uint8`. |
| **Internal Preprocessing** | $[0, 255] \to [-1.0, 1.0]$ | **Handled internally by the model** via an embedded `Rescaling(scale=1/127.5, offset=-1.0)` layer. External code only needs to supply standard $[0, 255]$ RGB pixel arrays. |

---

## 4. Class Mapping & Index Ordering

The model outputs a 5-element softmax probability vector with the following strict index mapping:

| Class Index | Class Name | Description & Typical Cargo Examples |
| :---: | :--- | :--- |
| **`0`** | **`box`** | Cardboard shipping boxes, cartons, crates, parcel packages. |
| **`1`** | **`chair`** | Office chairs, dining chairs, armchairs, wooden stools. |
| **`2`** | **`couch`** | Sofas, sectionals, living room couches, loveseats. |
| **`3`** | **`suitcase`** | Travel luggage, rolling suitcases, duffel bags, carry-ons. |
| **`4`** | **`table`** | Dining tables, desks, coffee tables, study tables. |

---

## 5. Loading the Model

```python
import tensorflow as tf

# Load the compiled Keras model
model_path = "models/cargo_mobilenetv2_best.keras"
model = tf.keras.models.load_model(model_path)
```

---

## 6. Output Format Schema

When calling inference on a single image, the returned response should follow this dictionary structure:

```json
{
  "predicted_class": "couch",
  "confidence": 0.6915,
  "probabilities": {
    "box": 0.0312,
    "chair": 0.1406,
    "couch": 0.6915,
    "suitcase": 0.0746,
    "table": 0.0621
  }
}
```

---

## 7. Drop-In Backend Integration Example (Python)

The following self-contained class can be directly imported and integrated into any backend API (e.g., FastAPI, Flask, Django, or AWS Lambda / GCP Cloud Run):

```python
import os
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf


class CargoClassifierService:
    """
    Drop-in Cargo Classification & Inference Service.
    Loads models/cargo_mobilenetv2_best.keras and classifies cargo images.
    """

    TARGET_CLASSES = ["box", "chair", "couch", "suitcase", "table"]
    IMAGE_SIZE = (128, 128)

    def __init__(self, model_path: str = "models/cargo_mobilenetv2_best.keras"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        
        # Load model once at service startup
        self.model = tf.keras.models.load_model(model_path)

    def preprocess(self, image_input: Any) -> np.ndarray:
        """
        Accepts a file path, file-like object, or PIL Image,
        and returns a normalized tensor of shape (1, 128, 128, 3).
        """
        if isinstance(image_input, str):
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            # File-like object (e.g., FastAPI UploadFile.file)
            img = Image.open(image_input)

        img_rgb = img.convert("RGB")
        img_resized = img_rgb.resize(self.IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Expand batch dimension: (1, 128, 128, 3)
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_input: Any) -> Dict[str, Any]:
        """
        Executes inference on a single cargo image.
        
        Returns:
            dict containing:
              - 'predicted_class': str
              - 'confidence': float
              - 'probabilities': Dict[str, float]
        """
        tensor = self.preprocess(image_input)
        probs = self.model.predict(tensor, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = self.TARGET_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        probabilities = {
            self.TARGET_CLASSES[i]: float(probs[i])
            for i in range(len(self.TARGET_CLASSES))
        }

        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
        }


# -----------------------------------------------------------------------------
# Quick Self-Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    classifier = CargoClassifierService(model_path="models/cargo_mobilenetv2_best.keras")
    
    # Test with a dummy image or real crop
    dummy_img = Image.new("RGB", (256, 256), color=(200, 150, 100))
    result = classifier.predict(dummy_img)
    print("Inference Test Result:", result)
```

---

## 8. Operational Scope & Important Non-Claims

> ### ⚠️ Critical Integration Notice for Product & Engineering Teams:
> 1. **Image Classification Only:** The model performs **2D semantic image categorization** (identifying *what* object is present).
> 2. **No Physical Weight Estimation:** Single 2D camera photos cannot measure weight. The model does **NOT** calculate kilograms ($kg$) or payload mass.
> 3. **No 3D Volumetric Measurements:** The model does **NOT** compute physical dimensions ($L \times W \times H\text{ in cm}$) or cubic meters ($m^3$) of cargo space.
> 4. **Heuristic Vehicle Recommendations:** Any vehicle recommendation derived from the model output is based on platform business rules mapped to cargo categories, not physical sensor measurements.

---

## 9. Current Benchmark & Evaluation Summary

Evaluated on the held-out test split using standard scikit-learn metrics:

| Metric | Score | Context |
| :--- | :---: | :--- |
| **Held-Out Test Accuracy** | **`63.64%`** ($14 / 22$) | Pre-trained MobileNetV2 ImageNet backbone. |
| **Held-Out Test Macro F1** | **`0.5562`** | Balanced average across all 5 classes. |
| **Held-Out Test Weighted F1** | **`0.6446`** | Weighted average reflecting class support. |
| **Best Validation Loss** | **`0.8757`** | Early stopping checkpoint (Epoch 7). |
| **Best Validation Accuracy** | **`68.42%`** | Validation split ($N=38$). |

### Per-Class Test Set Performance:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **`box`** | $1.0000$ | $0.4286$ | $0.6000$ | $7$ |
| **`chair`** | $0.8333$ | $0.6250$ | $0.7143$ | $8$ |
| **`couch`** | $0.6667$ | $1.0000$ | $0.8000$ | $2$ |
| **`suitcase`** | $0.0000$ | $0.0000$ | $0.0000$ | $1$ |
| **`table`** | $0.5000$ | $1.0000$ | $0.6667$ | $4$ |

---

## 10. Test Set Limitations & Prototype Evidence Notice

- **Dataset Size Notice:** The held-out test set contains **$22$ total crops** across the 5 classes.
- **Evidence Level:** These metrics demonstrate **prototype-level architectural feasibility** and prove the effectiveness of transfer learning over from-scratch CNNs.
- **Production Gate:** They do **NOT** constitute enterprise production validation. Before deploying to end customers, the model should be validated against a benchmark of $500+$ multi-angle, real-world customer uploads covering diverse packaging types, lighting conditions, and cluttered backgrounds.
