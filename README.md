# Cargo Vision CNN: Computer Vision for Logistics & Vehicle Suitability

> An end-to-end Computer Vision prototype that classifies cargo items into 5 operational categories (`box`, `chair`, `couch`, `suitcase`, `table`) and recommends suitable logistics vehicles to prevent vehicle-cargo mismatches in on-demand shipping platforms.

---

## 1. Problem Statement & Motivation

In on-demand freight and moving platforms, customers often book transport vehicles without accurately knowing whether their goods will fit. This leads to common real-world failure modes:
- A customer books a 3-wheeler or small pickup, but their cargo (e.g., large sofa or dining table) exceeds the payload space.
- The driver arrives, discovers the cargo is too bulky or fragile, and must cancel the trip.
- The platform suffers operational delays, wasted driver fuel, customer frustration, and high churn.

**Cargo Vision CNN** addresses this problem at the point of booking:
1. The customer uploads or snaps a photo of their cargo item.
2. The computer vision model classifies the cargo category.
3. The platform instantly suggests appropriate transport vehicles (e.g., Mini Tempo, Pickup, 14ft Flatbed, or Enclosed Moving Van) with handling guidance.

---

## 2. Benchmark Results & Model Comparison

We evaluated two distinct deep learning architectures on the exact same held-out test split ($N = 22$):

| Metric | Custom From-Scratch CNN Baseline | MobileNetV2 (Transfer Learning) | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Backbone** | 4 Conv Blocks + BN + GAP | Pre-trained ImageNet Feature Extractor | — |
| **Total Parameters** | $619,045$ | $2,423,749$ | $+1.80\text{M}$ |
| **Best Validation Loss** | $1.5658$ | **`0.8757`** | **$-0.6901$** |
| **Best Validation Accuracy** | $36.84\%$ | **`68.42%`** | **$+31.58\%$** |
| **Held-Out Test Accuracy** | $31.82\%$ ($7/22$) | **`63.64%`** ($14/22$) | **$+31.82\%$** |
| **Held-Out Test Macro F1** | $0.1000$ | **`0.5562`** | **$+0.4562$** |
| **Held-Out Test Weighted F1** | $0.1591$ | **`0.6446`** | **$+0.4855$** |
| **Training Duration (CPU)** | $142.32\text{ s}$ | **$44.10\text{ s}$** | $-98.22\text{ s}$ |

### Key Insight:
Training deep CNNs from scratch on compact datasets ($\approx 341$ training samples) suffers from rapid overfitting and class bias. Leveraging transfer learning via pre-trained **MobileNetV2** doubled overall test accuracy ($31.82\% \to 63.64\%$) and elevated macro F1 from $0.1000$ to $0.5562$.

---

## 3. Logistics Vehicle Recommendation Matrix

| Cargo Category | Primary Recommended Vehicle | Payload Capacity Class | Alternative Option | Handling & Securing Notes |
| :--- | :--- | :--- | :--- | :--- |
| **`box`** | **Mini Cargo Tempo / 3-Wheeler** *(Bajaj Maxima / Tata Ace Zip)* | Light Parcel / Box (Up to $500\text{ kg}$) | Pickup or 2-Wheeler (if single small parcel) | Stackable; requires tie-downs to prevent shifting during transit. |
| **`suitcase`** | **Cargo Auto / Hatchback / Sedan Luggage Carrier** | Personal Luggage (Up to $150\text{ kg}$) | Mini Van *(Maruti Eeco Cargo)* | Fragile handles and wheels; keep dry and avoid heavy top-loading. |
| **`chair`** | **Small Commercial Pickup** *(Tata Ace / Bolero Maxi Truck)* | Medium Furniture (Up to $750\text{ kg}$) | Flatbed Tempo | Protruding legs require corner padding and blanket wrapping. |
| **`table`** | **Light Commercial Truck / Flatbed** *(Tata 407 / Canter)* | Bulky Furniture (Up to $1.5\text{ Tons}$) | 14-ft Container Truck | Flat surface requires protective felt padding; load flat or dismantle legs. |
| **`couch`** | **Large Box Truck / Moving Van** *(Eicher 14ft - 19ft Container)* | Heavy Bulky Furniture (Up to $3.0\text{ Tons}$) | Enclosed 20ft Truck | High spatial volume; weatherproof enclosed carriage strictly recommended. |

> **DISCLAIMER:** Prototype vehicle recommendations are generated using category heuristics. Single 2D RGB camera photos cannot calculate exact physical weight (kg) or 3D volumetric dimensions (cm).

---

## 4. Repository Structure

```
cargo-vision-cnn/
├── data/
│   └── classification/
│       ├── README.md                      # Extraction statistics & documentation
│       ├── dataset_summary.json           # Audit provenance & split counts
│       ├── train/                         # 341 crops (5 classes)
│       ├── valid/                         # 38 crops (5 classes)
│       └── test/                          # 22 crops (5 classes)
├── models/
│   ├── cargo_cnn_best.keras               # Best from-scratch CNN checkpoint
│   └── cargo_mobilenetv2_best.keras       # Best MobileNetV2 checkpoint
├── notebooks/
│   └── cargo_vision_pipeline.ipynb        # Educational walkthrough notebook
├── results/
│   ├── data_inspection/                   # Contact sheets per class (box, chair, couch, etc.)
│   └── training/
│       ├── confusion_matrix.png           # From-scratch CNN confusion matrix
│       ├── training_curves.png            # From-scratch CNN learning curves
│       ├── training_history.csv           # From-scratch CNN epoch log
│       ├── training_summary.json          # From-scratch CNN metrics
│       ├── mobilenet_confusion_matrix.png # MobileNetV2 confusion matrix
│       ├── mobilenet_training_curves.png  # MobileNetV2 learning curves
│       ├── mobilenet_training_history.csv # MobileNetV2 epoch log
│       └── mobilenet_training_summary.json# MobileNetV2 metrics
├── scripts/
│   ├── dataset_config.py                  # Class mapping, augmentation & class weights
│   ├── inspect_classification_data.py     # Generates visual contact sheets
│   ├── model.py                           # Custom from-scratch CNN architecture definition
│   ├── predict_cargo.py                   # Single-image inference & vehicle recommender
│   ├── prepare_classification_data.py     # Extracts crops from YOLO annotations
│   ├── train.py                           # Training pipeline for from-scratch CNN
│   └── train_mobilenet.py                 # Training pipeline for MobileNetV2
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 5. Getting Started & Reproducibility Guide

### 5.1 Environment Setup
```powershell
# 1. Clone the repository
git clone https://github.com/<your-username>/cargo-vision-cnn.git
cd cargo-vision-cnn

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 5.2 Preprocess and Extract Dataset Crops
```powershell
# Dry-run inspection
python scripts/prepare_classification_data.py --dry-run

# Active crop extraction
python scripts/prepare_classification_data.py
```

### 5.3 Run Visual Sanity Check
```powershell
python scripts/inspect_classification_data.py
```

### 5.4 Train Models
```powershell
# Train Custom From-Scratch CNN Baseline
python scripts/train.py

# Train MobileNetV2 Transfer Learning Model
python scripts/train_mobilenet.py
```

### 5.5 Single-Image Prediction & Vehicle Recommendation
```powershell
# Run inference on any cargo image with MobileNetV2
python scripts/predict_cargo.py --image data/classification/test/couch/suggested-fIBXy0jYznvEfR66KMIj_jpg.rf.b1cca81a354c18d7f9a98bce11fef4d7_couch_0.jpg --model-type mobilenet

# Run inference with custom CNN baseline
python scripts/predict_cargo.py --image data/classification/test/chair/Chair_12_JPG.rf.6e9da998b38b475535da707cc6bef7f7_chair_0.jpg --model-type scratch
```

### 5.6 Launch Interactive Jupyter Walkthrough
```powershell
jupyter notebook notebooks/cargo_vision_pipeline.ipynb
```

---

## 6. Dataset Provenance & Attribution

- **Source Dataset:** Roboflow Universe Household Object Dataset
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Target Classes Extracted:** `Box` (ID 1), `Chair` (ID 2), `Couch` (ID 3), `Suitcase` (ID 11), `Table` (ID 12).
- **Split Preservation:** $100\%$ zero-leakage preservation of source train, validation, and test splits.

---

## 7. Technical Limitations & Production Roadmap

1. **Prototype Disclaimer:** The current model is an MVP designed to validate the computer vision to logistics recommendation pipeline. It is not currently deployed in production.
2. **2D Volumetric Limitation:** Single RGB images cannot determine exact depth, 3D volume, or weight. Future production systems will combine categorical predictions with ARCore/LiDAR bounding box spatial measurements.
3. **Dataset Scale:** Expanding dataset volume to $1,000+$ samples per class with diverse real-world lighting, loading bays, and truck interior backgrounds will further boost generalizability.
