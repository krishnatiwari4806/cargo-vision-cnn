# Physical Dimension Benchmark Dataset Protocol & Infrastructure

This directory houses the ground-truth physical benchmark dataset for evaluating the **Cargo Vision Dimension Estimation Engine** against the **$\ge 95.0\%$ Measurement Accuracy ($\text{MAPE} \le 5.0\%$)** target.

---

## 1. Directory Structure

```
data/dimension_benchmark/
├── README.md                 # Collection protocol, quality guidelines & schema documentation
├── images/                   # High-resolution original photographs (JPEG/PNG)
│   └── .gitkeep
├── annotations/              # 1-to-1 matching JSON ground-truth annotation files
│   └── .gitkeep
├── metadata/                 # Dataset-level metadata, instrument calibration records
│   └── .gitkeep
└── splits/                   # Object-level train/valid/test partition definitions
    └── .gitkeep
```

---

## 2. Physical Data Collection Protocol

Every sample in this benchmark dataset must represent a **real, physically measured cargo object**.

### 2.1 Required Equipment
1. **Measurement Instrument:**
   - Calibrated steel measuring tape (millimeter precision) **OR**
   - Digital laser distance meter (Class II laser, $\pm 1.5\text{ mm}$ accuracy).
2. **Reference Marker (Fiducial Anchor):**
   - Standard printable **ArUco Marker** (Dictionary: `DICT_4X4_50`, recommended size: $10.0\text{ cm} \times 10.0\text{ cm}$ square) **OR**
   - Standard **Credit / ID Card** (ISO/IEC 7810 ID-1 standard: $8.56\text{ cm} \times 5.398\text{ cm}$) **OR**
   - Standard **A4 Paper Sheet** ($21.0\text{ cm} \times 29.7\text{ cm}$).
3. **Camera Device:**
   - Standard smartphone or digital camera ($1080\text{p}$ to $4\text{K}$ resolution, natural lighting, without wide-angle fish-eye distortion).

---

### 2.2 Measurement Standards & Physical Definitions
- **Length ($L$):** Longest horizontal bounding extent of the object in centimeters ($cm$).
- **Width / Depth ($W$):** Perpendicular horizontal extent (thickness / depth) in centimeters ($cm$).
- **Height ($H$):** Maximum vertical extent from the ground contact plane to the highest point in centimeters ($cm$).
- **Weight ($W_{\text{kg}}$ - Optional):** Measured on an industrial floor scale or digital hanging luggage scale ($kg$).

```
                      +───────────────────────────+
                     /                           /│
                    /                           / │
                   +───────────────────────────+  │
                   │                           │  │  Height (H)
                   │                           │  │
                   │                           │  +
                   │                           │ /  Width (W)
                   │                           │/
                   +───────────────────────────+
                            Length (L)
```

---

### 2.3 Photography Rules
1. **Marker Placement:** Place the reference marker flat on or immediately beside the cargo object, aligned on the same depth plane ($Z$) facing the camera.
2. **Visibility:** The entire cargo object and the entire reference marker must be clearly visible and unoccluded in the frame.
3. **Lighting & Angle:** Ensure even lighting without harsh shadows across the marker corners.
4. **No Digital Manipulation:** Do not digitally crop, stretch, rotate, or alter the geometric aspect ratio of the captured image.

---

## 3. Annotation JSON Schema

For every image `images/<item_id>_<view>.jpg`, there must be an exact matching JSON file `annotations/<item_id>_<view>.json`.

### 3.1 Field-by-Field Specification

| Field Path | Type | Required | Description |
| :--- | :---: | :---: | :--- |
| `item_id` | `string` | **Yes** | Unique identifier for the physical object (e.g., `BENCH_001`). |
| `image_filename` | `string` | **Yes** | Base filename of the image in `images/` (e.g., `BENCH_001_front.jpg`). |
| `object_category` | `string` | **Yes** | Cargo category (`box`, `chair`, `couch`, `table`, `suitcase`, `car`, etc.). |
| `image.width_px` | `integer` | **Yes** | Image width in pixels ($> 0$). |
| `image.height_px` | `integer` | **Yes** | Image height in pixels ($> 0$). |
| `reference.present` | `boolean` | **Yes** | Whether a reference marker is present in the image. |
| `reference.type` | `string` | If marker present | Marker type (`aruco_4x4_50`, `credit_card_id1`, `a4_sheet`, `none`). |
| `reference.size_cm` | `float` | If marker present | Side length / primary dimension of the marker in $cm$ ($> 0$). |
| `ground_truth.length_cm` | `float` | **Yes** | Physically measured length in centimeters ($> 0$). |
| `ground_truth.width_cm` | `float` | **Yes** | Physically measured width in centimeters ($> 0$). |
| `ground_truth.height_cm` | `float` | **Yes** | Physically measured height in centimeters ($> 0$). |
| `ground_truth.weight_kg` | `float` or `null` | Optional | Physically measured weight in $kg$ ($> 0$ if provided). |
| `measurement_method` | `string` | **Yes** | Instrument used (`measuring_tape`, `laser_meter`, `caliper`). |
| `camera.distance_m` | `float` or `null` | Optional | Estimated or laser-measured distance from camera to object ($m$). |
| `camera.view_angle_deg` | `float` or `null` | Optional | Approximate camera angle relative to frontal plane ($0^\circ = \text{frontal}$). |

---

### 3.2 Canonical JSON Template (`annotations/BENCH_001_front.json`)

```json
{
  "item_id": "BENCH_001",
  "image_filename": "BENCH_001_front.jpg",
  "object_category": "box",
  "image": {
    "width_px": 1920,
    "height_px": 1080
  },
  "reference": {
    "present": true,
    "type": "aruco_4x4_50",
    "size_cm": 10.0
  },
  "ground_truth": {
    "length_cm": 45.2,
    "width_cm": 35.0,
    "height_cm": 29.8,
    "weight_kg": 8.4
  },
  "measurement_method": "measuring_tape",
  "camera": {
    "distance_m": 1.85,
    "view_angle_deg": 0.0
  }
}
```

---

## 4. Data Quality & Acceptance Rules

1. **No Synthetic or Guessed Data:** Never invent dimensions or copy catalog numbers. All values must be measured with a physical tool.
2. **Quantity Separation:** Quantity ($Q$) is a transactional operational input and must **never** be multiplied into unit ground-truth dimensions.
3. **Independent Validation Split Isolation:**
   - Dataset splitting is performed **strictly by physical object (`item_id`)**, never by image.
   - If an object `BENCH_001` has 3 photographs (front, side, angled), **all 3 images must belong to the same split**.
   - No physical object in the training/development split may ever appear in the test split.
4. **Target Dataset Size:**
   - **Minimum Feasibility Threshold:** $50$ real physical samples.
   - **Preferred Enterprise Target:** $100+$ physical samples covering diverse categories (`box`, `chair`, `couch`, `table`, `suitcase`, `car`, `appliances`, `pallets`).

---

## 5. Dataset Validation Utility

A validation script is provided to enforce zero-defect integrity before any benchmark execution:

```powershell
python scripts/validate_dimension_dataset.py
```
Checks:
- All required schema fields are present and properly typed.
- Matching image exists in `images/` for every annotation in `annotations/`.
- All dimensions ($L, W, H$) are positive non-zero floats.
- No duplicate `item_id`s or orphaned image/annotation pairs.
