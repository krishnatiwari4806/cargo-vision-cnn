"""
inspect_classification_data.py
==============================
Project: Cargo Vision CNN
Purpose: Visual sanity check and integrity audit of generated classification crops.
         Generates 5 contact sheets in results/data_inspection/ (one per target class)
         displaying randomly sampled crops, filenames, and image dimensions.
"""

import os
import glob
import random
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

TARGET_CLASSES = ["box", "chair", "couch", "suitcase", "table"]
DATA_DIR = "data/classification/train"
OUTPUT_DIR = "results/data_inspection"
NUM_SAMPLES_PER_CLASS = 5
RANDOM_SEED = 42

TILE_SIZE = 220
LABEL_HEIGHT = 45
HEADER_HEIGHT = 40
PADDING = 15


def create_contact_sheet(
    class_name: str,
    sampled_files: List[str],
    output_path: str,
) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Creates a visual contact sheet image for a single class and verifies images.
    Returns: (is_valid, list of (width, height) dimensions)
    """
    n_images = len(sampled_files)
    if n_images == 0:
        return False, []

    dimensions = []
    loaded_tiles = []

    for fpath in sampled_files:
        # 1. Verification checks
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            return False, []

        with Image.open(fpath) as img:
            img.verify()

        with Image.open(fpath) as img:
            rgb_img = img.convert("RGB")
            w, h = rgb_img.size
            dimensions.append((w, h))

            # Resize with aspect ratio preserved to fit TILE_SIZE x TILE_SIZE
            aspect = w / h
            if aspect > 1.0:
                new_w = TILE_SIZE
                new_h = max(1, int(TILE_SIZE / aspect))
            else:
                new_h = TILE_SIZE
                new_w = max(1, int(TILE_SIZE * aspect))

            resized = rgb_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Create standard tile background
            tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), color=(240, 240, 240))
            paste_x = (TILE_SIZE - new_w) // 2
            paste_y = (TILE_SIZE - new_h) // 2
            tile.paste(resized, (paste_x, paste_y))

            fname = os.path.basename(fpath)
            loaded_tiles.append((tile, fname, w, h))

    # Canvas Dimensions (1 row of n_images tiles)
    canvas_w = (TILE_SIZE * n_images) + (PADDING * (n_images + 1))
    canvas_h = HEADER_HEIGHT + TILE_SIZE + LABEL_HEIGHT + (PADDING * 2)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Use default bitmap font
    font = ImageFont.load_default()

    # Draw Header Banner
    header_text = f"Class: {class_name.upper()}  |  Train Split Sample Inspection  |  {n_images} Random Crops"
    draw.rectangle([(0, 0), (canvas_w, HEADER_HEIGHT)], fill=(30, 41, 59))
    draw.text((PADDING, 14), header_text, fill=(255, 255, 255), font=font)

    # Draw Tiles and Labels
    for idx, (tile, fname, orig_w, orig_h) in enumerate(loaded_tiles):
        x = PADDING + idx * (TILE_SIZE + PADDING)
        y = HEADER_HEIGHT + PADDING

        # Paste tile with border
        canvas.paste(tile, (x, y))
        draw.rectangle([(x, y), (x + TILE_SIZE, y + TILE_SIZE)], outline=(200, 200, 200), width=1)

        # Label texts
        # Truncate filename if too long
        display_name = fname if len(fname) <= 26 else f"{fname[:12]}...{fname[-10:]}"
        dim_text = f"{orig_w}x{orig_h} px"

        label_y = y + TILE_SIZE + 6
        draw.text((x + 2, label_y), display_name, fill=(15, 23, 42), font=font)
        draw.text((x + 2, label_y + 16), dim_text, fill=(100, 116, 139), font=font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path, "JPEG", quality=95)
    return True, dimensions


def main():
    print("=" * 70)
    print("Cargo Vision CNN - Classification Data Sanity Inspection")
    print(f"Sampling up to {NUM_SAMPLES_PER_CLASS} crops/class from: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 70)

    random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_stats = []
    all_valid = True

    for class_name in TARGET_CLASSES:
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"[Error] Class directory not found: {class_dir}")
            all_valid = False
            continue

        crop_files = sorted(glob.glob(os.path.join(class_dir, "*.jpg")))
        total_crops = len(crop_files)

        if total_crops == 0:
            print(f"[Error] No crops found for class '{class_name}'.")
            all_valid = False
            continue

        # Deterministic sample
        sampled = random.sample(crop_files, min(NUM_SAMPLES_PER_CLASS, total_crops))

        out_contact_sheet = os.path.join(OUTPUT_DIR, f"{class_name}.jpg")
        success, dims = create_contact_sheet(class_name, sampled, out_contact_sheet)

        if not success:
            print(f"[Error] Failed to generate contact sheet for '{class_name}'.")
            all_valid = False
            continue

        widths = [d[0] for d in dims]
        heights = [d[1] for d in dims]

        summary_stats.append({
            "class": class_name,
            "total_crops": total_crops,
            "sampled": len(sampled),
            "min_dim": f"{min(widths)}x{min(heights)}",
            "max_dim": f"{max(widths)}x{max(heights)}",
            "avg_w": int(sum(widths) / len(widths)),
            "avg_h": int(sum(heights) / len(heights)),
            "sheet_path": out_contact_sheet,
        })

    # Print Summary Table
    print("\n" + "=" * 70)
    print(f"{'Class':<10} | {'Total':<6} | {'Sampled':<8} | {'Min Dim':<12} | {'Max Dim':<12} | {'Avg Dim':<10} | {'Contact Sheet':<15}")
    print("-" * 70)
    for st in summary_stats:
        avg_dim = f"{st['avg_w']}x{st['avg_h']}"
        sheet_fname = os.path.basename(st['sheet_path'])
        print(f"{st['class']:<10} | {st['total_crops']:<6} | {st['sampled']:<8} | {st['min_dim']:<12} | {st['max_dim']:<12} | {avg_dim:<10} | {sheet_fname:<15}")
    print("=" * 70)

    if all_valid:
        print("\n[PASSED] Visual sanity check complete: 5 contact sheets generated, 0 corrupt crops.")
    else:
        print("\n[FAILED] Visual sanity check encountered errors.")


if __name__ == "__main__":
    main()
