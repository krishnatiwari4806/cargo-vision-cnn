#!/usr/bin/env python3
"""
run.py - Simple Interactive Cargo Image Test Runner
===================================================
Project: Cargo Vision Logistics System
Purpose: Provides a streamlined, interactive CLI for testing cargo images without
         requiring deep learning arguments, manual file paths, or CLI flags.

Workflow:
  1. Automatically scans 'data/test_images/' for supported images (.jpg, .jpeg, .png, .webp, .bmp).
  2. Auto-selects single image or prompts user for selection from multiple images.
  3. Prompts minimal guided questions: Cargo Category (or Auto-detect), Quantity, and Optional Weight.
  4. Executes the production CargoAnalysisPipeline.
  5. Formats and displays a clean, transparent terminal analysis report.

Usage:
  .\\.venv\\Scripts\\python.exe run.py
  or
  test.bat
"""

import os
import sys
from typing import List, Optional, Tuple, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.dimension_estimator import DEFAULT_CATEGORY_PRIORS
from scripts.cargo_analysis_pipeline import CargoAnalysisPipeline

DEFAULT_TEST_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "test_images")
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Canonical categories in display order
CANONICAL_CATEGORIES: List[str] = [
    "box",
    "chair",
    "couch",
    "table",
    "suitcase",
    "car",
    "refrigerator",
    "tv",
    "bed",
    "desk",
]


def find_test_images(test_dir: str = DEFAULT_TEST_IMAGE_DIR) -> List[str]:
    """
    Scans the test images folder and returns absolute paths of supported image files.
    """
    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)
        return []

    found = []
    for entry in sorted(os.listdir(test_dir)):
        ext = os.path.splitext(entry)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            full_path = os.path.join(test_dir, entry)
            if os.path.isfile(full_path):
                found.append(full_path)
    return found


def prompt_image_selection(images: List[str], input_fn=input) -> Optional[str]:
    """
    Guides the user to select an image from the available list.
    If 0 images: prints instructions and returns None.
    If 1 image: auto-selects and returns path.
    If >1 images: displays numbered menu.
    """
    if not images:
        print("\n" + "=" * 60)
        print("                 CARGO VISION IMAGE TESTER")
        print("=" * 60)
        print(f"No test image found in: data/test_images/\n")
        print("Quick Setup:")
        print("  1. Place a test cargo image (.jpg, .png, .jpeg, .webp, .bmp)")
        print(f"     into the folder: data/test_images/")
        print("  2. Then run this runner again:")
        print(r"     .\.venv\Scripts\python.exe run.py")
        print("=" * 60 + "\n")
        return None

    if len(images) == 1:
        selected = images[0]
        print(f"\n[Auto-Selected Image]: {os.path.basename(selected)}")
        return selected

    print("\n" + "=" * 60)
    print("                 SELECT TEST IMAGE")
    print("=" * 60)
    for idx, path in enumerate(images, start=1):
        print(f"  {idx}. {os.path.basename(path)}")
    print("=" * 60)

    while True:
        try:
            choice_str = input_fn(f"Select image (1-{len(images)}) [default: 1]: ").strip()
            if choice_str == "":
                return images[0]
            choice_num = int(choice_str)
            if 1 <= choice_num <= len(images):
                return images[choice_num - 1]
            print(f"Please enter a valid number between 1 and {len(images)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def prompt_cargo_category(input_fn=input) -> Tuple[Optional[str], str]:
    """
    Prompts the user for cargo category or Auto-detect.
    Returns (canonical_category_slug_or_None, display_name).
    """
    print("\nWhat cargo is this?")
    print("  0. Auto detect (use AI Vision model)")
    for idx, cat_slug in enumerate(CANONICAL_CATEGORIES, start=1):
        prior_info = DEFAULT_CATEGORY_PRIORS.get(cat_slug, {})
        desc = prior_info.get("description", cat_slug.capitalize())
        print(f"  {idx}. {cat_slug.capitalize()} ({desc})")

    while True:
        try:
            choice_str = input_fn(f"Select option (0-{len(CANONICAL_CATEGORIES)}) [default: 0]: ").strip()
            if choice_str == "" or choice_str == "0":
                return None, "Auto detect"
            choice_num = int(choice_str)
            if 1 <= choice_num <= len(CANONICAL_CATEGORIES):
                selected_cat = CANONICAL_CATEGORIES[choice_num - 1]
                return selected_cat, selected_cat.capitalize()
            print(f"Please enter a number between 0 and {len(CANONICAL_CATEGORIES)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def prompt_quantity(input_fn=input) -> int:
    """
    Prompts the user for cargo quantity (integer >= 1). Default: 1.
    """
    while True:
        try:
            qty_str = input_fn("Quantity? [default: 1]: ").strip()
            if qty_str == "":
                return 1
            qty = int(qty_str)
            if qty >= 1:
                return qty
            print("Quantity must be a positive integer (>= 1).")
        except ValueError:
            print("Invalid input. Please enter an integer.")


def prompt_user_weight(input_fn=input) -> Optional[float]:
    """
    Prompts whether user knows the actual physical weight.
    If Yes: prompts for weight per item in kg.
    If No: returns None.
    """
    print("\nDo you know the actual weight?")
    print("  1. Yes")
    print("  2. No")

    while True:
        choice_str = input_fn("Select option (1-2) [default: 2]: ").strip()
        if choice_str == "" or choice_str == "2":
            return None
        if choice_str == "1":
            break
        print("Please enter 1 or 2.")

    while True:
        try:
            wt_str = input_fn("Enter weight per item in kg: ").strip()
            wt_val = float(wt_str)
            if wt_val > 0:
                return round(wt_val, 2)
            print("Weight must be greater than 0 kg.")
        except ValueError:
            print("Invalid weight. Please enter a positive number (e.g. 15.5).")


def format_analysis_result(result: Dict[str, Any], user_declared_cargo: Optional[str] = None) -> str:
    """
    Formats the analysis result dictionary into a clean, human-readable terminal report.
    """
    if result.get("status") == "ERROR":
        lines = [
            "\n" + "=" * 60,
            "                   CARGO ANALYSIS ERROR",
            "=" * 60,
            f"Image  : {os.path.basename(result.get('input_image', 'Unknown'))}",
            f"Error  : {result.get('error', 'Unknown pipeline failure')}",
        ]
        if result.get("warnings"):
            lines.append("\nWarnings:")
            for w in result["warnings"]:
                lines.append(f"  - {w}")
        lines.append("=" * 60 + "\n")
        return "\n".join(lines)

    img_name = os.path.basename(result.get("input_image", "Unknown"))
    classification = result.get("classification", {})
    det_class = classification.get("class_name", "Unknown").capitalize()
    det_conf = classification.get("confidence", 0.0)
    reliability = result.get("reliability", "UNKNOWN")

    # Dimensions & volume
    dims = result.get("dimensions", {})
    l = dims.get("length_cm")
    w = dims.get("width_cm")
    h = dims.get("height_cm")
    dim_src = dims.get("source", "category_prior")
    dim_str = f"{l:.1f} × {w:.1f} × {h:.1f} cm" if (l and w and h) else "Unavailable"

    # Cargo summary
    c_summary = result.get("cargo_summary", {})
    qty = result.get("quantity", 1)
    unit_vol = c_summary.get("unit_volume_m3")
    total_vol = c_summary.get("total_volume_m3")
    floor_area = c_summary.get("required_floor_area_m2")

    unit_wt = c_summary.get("unit_weight_kg")
    total_wt = c_summary.get("total_weight_kg")
    wt_src = c_summary.get("weight_source", "category_prior").upper()
    wt_status = c_summary.get("weight_status", "ESTIMATED")
    wt_unc_pct = dims.get("weight_uncertainty_percent")
    wt_unc_kg = dims.get("weight_uncertainty_kg")

    unc_str = ""
    if wt_unc_pct is not None and wt_unc_pct > 0:
        unc_str = f"±{wt_unc_pct:.1f}%"
        if wt_unc_kg is not None:
            unc_str += f" (±{wt_unc_kg:.1f} kg)"
    elif wt_src == "USER_PROVIDED":
        unc_str = "Declared (Exact)"
    else:
        unc_str = "N/A"

    # Vehicle Recommendation
    v_rec = result.get("vehicle_recommendation", {})
    v_name = v_rec.get("vehicle_name", "None")
    v_reason = v_rec.get("reason", "N/A")
    v_alts = v_rec.get("alternatives", [])
    v_alts_str = ", ".join(v_alts) if v_alts else "None"

    # Declared vs Detected header
    cargo_display = det_class
    if user_declared_cargo:
        cargo_display = f"{user_declared_cargo.capitalize()} (User-Declared)"
    else:
        cargo_display = f"{det_class} (Auto-Detected)"

    report_lines = [
        "\n" + "=" * 60,
        "                  CARGO ANALYSIS RESULT",
        "=" * 60,
        f"Image               : {img_name}",
        f"Cargo               : {cargo_display}",
        f"Quantity            : {qty}",
        "",
        "--- Detection & Identification ---",
        f"Model Detection     : {det_class}",
        f"Confidence Score    : {det_conf * 100:.1f}%",
        f"Reliability Status  : {reliability}",
        "",
        "--- Physical Dimensions & Volume ---",
        f"Dimensions / Item   : {dim_str} ({dim_src})",
        f"Volume / Item       : {unit_vol:.4f} m³" if unit_vol else "Volume / Item       : N/A",
        f"Total Usable Volume : {total_vol:.4f} m³" if total_vol else "Total Usable Volume : N/A",
        f"Required Bed Area   : {floor_area:.3f} m²" if floor_area else "Required Bed Area   : N/A",
        "",
        "--- Weight Estimation ---",
        f"Weight / Item       : {unit_wt:.1f} kg" if unit_wt else "Weight / Item       : N/A",
        f"Weight Source       : {wt_src}",
        f"Weight Status       : {wt_status}",
        f"Weight Uncertainty  : {unc_str}",
        f"Total Cargo Weight  : {total_wt:.1f} kg" if total_wt else "Total Cargo Weight  : N/A",
        "",
        "--- Vehicle Recommendation ---",
        f"Recommended Vehicle : {v_name}",
        f"Reason              : {v_reason}",
        f"Alternative(s)      : {v_alts_str}",
    ]

    warnings = result.get("warnings", [])
    if warnings:
        report_lines.append("")
        report_lines.append("--- Disclaimers & Operational Warnings ---")
        for w in warnings:
            report_lines.append(f"  • {w}")

    report_lines.append("=" * 60 + "\n")
    return "\n".join(report_lines)


def run_interactive_test(test_dir: str = DEFAULT_TEST_IMAGE_DIR, input_fn=input) -> int:
    """
    Main interactive entry point orchestrating the test flow.
    """
    print("\n" + "=" * 60)
    print("          CARGO VISION — INTERACTIVE TEST RUNNER")
    print("=" * 60)

    # 1. Scan for test images
    images = find_test_images(test_dir)
    selected_image = prompt_image_selection(images, input_fn=input_fn)
    if selected_image is None:
        return 0

    # 2. Header
    print("\n" + "=" * 60)
    print("                   CARGO VISION TEST")
    print("=" * 60)
    print(f"Image: {os.path.basename(selected_image)}")

    # 3. Interactive questions
    cat_slug, cat_display = prompt_cargo_category(input_fn=input_fn)
    qty = prompt_quantity(input_fn=input_fn)
    user_weight = prompt_user_weight(input_fn=input_fn)

    print("\n[Processing image with Cargo Vision Pipeline...]")

    try:
        pipeline = CargoAnalysisPipeline()
        result = pipeline.analyze(
            image_path=selected_image,
            quantity=qty,
            user_weight_kg=user_weight,
            category_override=cat_slug,
        )
        report = format_analysis_result(result, user_declared_cargo=cat_slug)
        print(report)
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print("                  UNEXPECTED PIPELINE ERROR")
        print("=" * 60)
        print(f"An error occurred while analyzing '{os.path.basename(selected_image)}':")
        print(f"  {str(e)}")
        print("=" * 60 + "\n")
        return 1


def main():
    try:
        exit_code = run_interactive_test()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[Session cancelled by user.]")
        sys.exit(0)


if __name__ == "__main__":
    main()
