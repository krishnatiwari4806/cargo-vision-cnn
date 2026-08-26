# Cargo Vision CNN: Classification Dataset Report

This directory contains the 5-class object-crop classification dataset extracted from `deployment_dataset/`.

## 1. Dataset Overview
- **Source Dataset:** `deployment_dataset/` (Roboflow Universe / Household Item Dataset, License: CC BY 4.0)
- **Target Classes:**
  - `box` (Class ID: 1)
  - `chair` (Class ID: 2)
  - `couch` (Class ID: 3)
  - `suitcase` (Class ID: 11)
  - `table` (Class ID: 12)
- **Padding Ratio:** 8.0% contextual margin around bounding box.
- **Data Leakage Safeguard:** 100% preservation of original splits (`train`, `valid`, `test`).
- **Total Crops Generated:** 473
- **Generated At (UTC):** 2026-08-26T17:15:00Z

## 2. Crop Distribution by Class and Split

| Class Name | Train Crops | Valid Crops | Test Crops | Total Crops |
| :--- | :---: | :---: | :---: | :---: |
| **box** | 59 | 9 | 7 | **75** |
| **chair** | 151 | 13 | 8 | **172** |
| **couch** | 49 | 4 | 2 | **55** |
| **suitcase** | 21 | 4 | 1 | **26** |
| **table** | 111 | 19 | 15 | **145** |
| **TOTAL** | **391** | **49** | **33** | **473** |

## 3. Annotation & Parsing Statistics
- **Total Target Annotations Encountered:** 473
- **Polygon Segmentation Annotations:** 465
- **Standard Bounding Box Annotations:** 8
- **Rejected / Invalid Annotations:** 0
- **Verified Valid Crops Written:** 473

## 4. Class Imbalance & Training Recommendations
- `chair` (172 crops) has the highest representation.
- `suitcase` (26 crops) has the lowest representation.
- **Recommendation:** Utilize on-the-fly data augmentation (random horizontal flips, slight rotations, zooming) during CNN training to provide robust feature generalization.
