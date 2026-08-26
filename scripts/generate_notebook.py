"""
generate_notebook.py
Generates the comprehensive educational Jupyter Notebook for Cargo Vision CNN.
"""

import json
import os

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Cargo Vision CNN: Computer Vision for Logistics & Vehicle Suitability\n",
                "\n",
                "## 1. Project Background & Problem Framing\n",
                "In on-demand freight and logistics platforms, customers frequently book vehicles without accurately estimating their cargo size or vehicle requirements. When goods are oversized or mismatched for the selected vehicle, bookings fail, drivers face delays, and logistics operations suffer high churn.\n",
                "\n",
                "### Goal of this Prototype:\n",
                "1. Build an end-to-end computer vision classification pipeline for **5 core cargo categories** (`box`, `chair`, `couch`, `suitcase`, `table`).\n",
                "2. Implement a **custom from-scratch CNN** to study deep feature learning in low-data regimes.\n",
                "3. Implement a **transfer learning baseline (MobileNetV2)** to demonstrate the benefits of pre-trained spatial representations.\n",
                "4. Bridge computer vision predictions to **logistics business logic** via a rule-based vehicle suitability recommendation engine.\n",
                "\n",
                "> **Disclaimer:** This prototype uses single 2D RGB photos to classify cargo category. Single 2D images cannot compute exact physical weight (kg) or 3D volume (cm)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Environment Setup & Dependency Imports"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import json\n",
                "import glob\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "from PIL import Image\n",
                "import tensorflow as tf\n",
                "from sklearn.metrics import classification_report, confusion_matrix\n",
                "\n",
                "# Add project root to sys.path\n",
                "sys.path.insert(0, os.path.abspath('..'))\n",
                "\n",
                "from scripts.dataset_config import (\n",
                "    TARGET_CLASSES,\n",
                "    INDEX_TO_CLASS,\n",
                "    IMAGE_SIZE,\n",
                "    compute_train_class_weights,\n",
                "    get_all_split_counts,\n",
                ")\n",
                "from scripts.predict_cargo import VEHICLE_CATALOG\n",
                "\n",
                "print(f\"TensorFlow Version: {tf.__version__}\")\n",
                "print(f\"Target Cargo Classes: {TARGET_CLASSES}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Dataset Audit & Preprocessing Summary\n",
                "The classification dataset was extracted from an audited multi-object household dataset (`deployment_dataset/`, CC BY 4.0). Segmentations and bounding boxes for the 5 target classes were isolated with 8% contextual padding to generate 401 high-quality crops while strictly preserving the original train/valid/test splits."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open('../data/classification/dataset_summary.json', 'r') as f:\n",
                "    summary = json.load(f)\n",
                "\n",
                "df_counts = pd.DataFrame(summary['crops_per_class_per_split']).T\n",
                "df_counts['Total'] = df_counts.sum(axis=1)\n",
                "display(df_counts)\n",
                "\n",
                "# Compute and display balanced class weights\n",
                "weights = compute_train_class_weights()\n",
                "print(\"\\nComputed Balanced Class Weights (Train Split):\")\n",
                "for idx, w in weights.items():\n",
                "    print(f\"  - {TARGET_CLASSES[idx]:<10}: {w:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Visual Inspection of Generated Crops\n",
                "Let's inspect sample crops across the 5 target cargo classes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))\n",
                "\n",
                "for idx, cname in enumerate(TARGET_CLASSES):\n",
                "    sample_path = glob.glob(f'../data/classification/train/{cname}/*.jpg')[0]\n",
                "    img = Image.open(sample_path)\n",
                "    axes[idx].imshow(img)\n",
                "    axes[idx].set_title(f\"{cname.upper()}\\n{img.size[0]}x{img.size[1]} px\", fontweight='bold')\n",
                "    axes[idx].axis('off')\n",
                "\n",
                "plt.suptitle(\"Sample Cargo Crops from Train Split\", fontsize=14, fontweight='bold', y=1.05)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Architectural Comparison: From-Scratch CNN vs. MobileNetV2 Transfer Learning\n",
                "\n",
                "We benchmark two distinct paradigms on the exact same held-out test split ($N=22$):\n",
                "1. **Custom From-Scratch CNN:** 4 convolutional blocks + BatchNorm + Dropout + GlobalAveragePooling2D ($619,045$ parameters).\n",
                "2. **MobileNetV2 (Transfer Learning):** Pretrained ImageNet feature extractor with frozen weights + classification head."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open('../results/training/training_summary.json', 'r') as f:\n",
                "    scratch_res = json.load(f)\n",
                "\n",
                "with open('../results/training/mobilenet_training_summary.json', 'r') as f:\n",
                "    mobilenet_res = json.load(f)\n",
                "\n",
                "benchmark_df = pd.DataFrame([\n",
                "    {\n",
                "        'Architecture': 'Custom From-Scratch CNN',\n",
                "        'Best Val Loss': scratch_res['best_val_loss'],\n",
                "        'Best Val Acc': f\"{scratch_res['best_val_accuracy']*100:.2f}%\",\n",
                "        'Test Accuracy': f\"{scratch_res['test_accuracy']*100:.2f}%\",\n",
                "        'Macro F1': scratch_res['test_macro_f1'],\n",
                "        'Weighted F1': scratch_res['test_weighted_f1'],\n",
                "    },\n",
                "    {\n",
                "        'Architecture': 'MobileNetV2 (Transfer Learning)',\n",
                "        'Best Val Loss': mobilenet_res['best_val_loss'],\n",
                "        'Best Val Acc': f\"{mobilenet_res['best_val_accuracy']*100:.2f}%\",\n",
                "        'Test Accuracy': f\"{mobilenet_res['test_accuracy']*100:.2f}%\",\n",
                "        'Macro F1': mobilenet_res['test_macro_f1'],\n",
                "        'Weighted F1': mobilenet_res['test_weighted_f1'],\n",
                "    },\n",
                "])\n",
                "\n",
                "display(benchmark_df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Confusion Matrix & Diagnostic Visualizations"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))\n",
                "\n",
                "# From-Scratch Confusion Matrix\n",
                "cm_scratch = np.array(scratch_res['confusion_matrix'])\n",
                "im1 = ax1.imshow(cm_scratch, cmap=plt.cm.Blues)\n",
                "ax1.set_title(\"Custom From-Scratch CNN (31.82% Acc)\", fontweight='bold', fontsize=12)\n",
                "ax1.set_xticks(range(len(TARGET_CLASSES)))\n",
                "ax1.set_yticks(range(len(TARGET_CLASSES)))\n",
                "ax1.set_xticklabels(TARGET_CLASSES, rotation=30, ha='right')\n",
                "ax1.set_yticklabels(TARGET_CLASSES)\n",
                "for i in range(5):\n",
                "    for j in range(5):\n",
                "        ax1.text(j, i, str(cm_scratch[i, j]), ha='center', va='center', fontweight='bold',\n",
                "                 color='white' if cm_scratch[i, j] > cm_scratch.max()/2 else 'black')\n",
                "\n",
                "# MobileNetV2 Confusion Matrix\n",
                "cm_mobilenet = np.array(mobilenet_res['confusion_matrix'])\n",
                "im2 = ax2.imshow(cm_mobilenet, cmap=plt.cm.Greens)\n",
                "ax2.set_title(\"MobileNetV2 Transfer Learning (63.64% Acc)\", fontweight='bold', fontsize=12)\n",
                "ax2.set_xticks(range(len(TARGET_CLASSES)))\n",
                "ax2.set_yticks(range(len(TARGET_CLASSES)))\n",
                "ax2.set_xticklabels(TARGET_CLASSES, rotation=30, ha='right')\n",
                "ax2.set_yticklabels(TARGET_CLASSES)\n",
                "for i in range(5):\n",
                "    for j in range(5):\n",
                "        ax2.text(j, i, str(cm_mobilenet[i, j]), ha='center', va='center', fontweight='bold',\n",
                "                 color='white' if cm_mobilenet[i, j] > cm_mobilenet.max()/2 else 'black')\n",
                "\n",
                "plt.suptitle(\"Test Set Confusion Matrix Comparison (N=22)\", fontsize=14, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Interactive Prediction & Vehicle Suitability Recommendation\n",
                "Let's test single-image inference and trigger the vehicle recommendation engine."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load best MobileNetV2 model\n",
                "model = tf.keras.models.load_model('../models/cargo_mobilenetv2_best.keras')\n",
                "\n",
                "test_image_path = glob.glob('../data/classification/test/couch/*.jpg')[0]\n",
                "img = Image.open(test_image_path).convert('RGB')\n",
                "img_resized = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)\n",
                "tensor = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)\n",
                "\n",
                "probs = model.predict(tensor, verbose=0)[0]\n",
                "pred_idx = int(np.argmax(probs))\n",
                "pred_class = INDEX_TO_CLASS[pred_idx]\n",
                "confidence = float(probs[pred_idx])\n",
                "\n",
                "# Display\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
                "ax1.imshow(img)\n",
                "ax1.set_title(f\"Input Image\\nTrue: COUCH | Pred: {pred_class.upper()} ({confidence*100:.1f}%)\", fontweight='bold')\n",
                "ax1.axis('off')\n",
                "\n",
                "bars = ax2.barh(TARGET_CLASSES, probs * 100, color='#3b82f6')\n",
                "bars[pred_idx].set_color('#10b981')\n",
                "ax2.set_xlabel('Probability (%)')\n",
                "ax2.set_xlim(0, 100)\n",
                "ax2.set_title('Softmax Prediction Probabilities', fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Vehicle Recommendation Output\n",
                "rec = VEHICLE_CATALOG[pred_class]\n",
                "print(\"=\" * 70)\n",
                "print(\"LOGISTICS VEHICLE RECOMMENDATION\")\n",
                "print(\"=\" * 70)\n",
                "print(f\"Recommended Vehicle:  {rec['primary_vehicle']}\")\n",
                "print(f\"Capacity Class:       {rec['capacity_class']}\")\n",
                "print(f\"Alternative Vehicle:  {rec['alternative_vehicle']}\")\n",
                "print(f\"Handling Notes:       {rec['handling_notes']}\")\n",
                "print(\"=\" * 70)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Summary & Next Steps\n",
                "\n",
                "### Key Takeaways:\n",
                "1. **From-Scratch Deep CNNs in Small Regimes:** Deep CNNs trained from scratch on raw pixels easily overfit when training data is under 100 samples per class.\n",
                "2. **Transfer Learning Superiority:** Pretrained ImageNet features (MobileNetV2) elevated accuracy from **31.82% to 63.64%** with an F1 score improvement from **0.1000 to 0.5562**.\n",
                "3. **Logistics Domain Alignment:** Categorical predictions successfully mapped to operational vehicle requirements with explicit safety disclaimers.\n",
                "\n",
                "### Production Roadmap:\n",
                "- Expand dataset to 1,000+ annotations per class covering diverse packaging and indoor/outdoor lighting.\n",
                "- Incorporate object detection / segmentation (e.g. YOLOv8) to localize cargo in cluttered multi-item environments.\n",
                "- Integrate multi-view or depth sensors (LiDAR / ARCore) if exact volumetric calculations are required in future iterations."
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.12"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

nb_path = "notebooks/cargo_vision_pipeline.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully generated notebook at: {nb_path}")
