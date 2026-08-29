"""
monitor_training.py - Non-Intrusive Live Training Progress Monitor
==================================================================
Project: Cargo Vision Logistics System
Purpose: Reads and displays the genuine live YOLOv8n training progress, losses, and validation
         metrics directly to the terminal without modifying or interrupting the running training process.
"""

import os
import sys
import glob
import re

LOG_FILE = r"C:\Users\ACER\.gemini\antigravity-ide\brain\002f8397-d358-41e1-912c-bbc6a4ace65c\.system_generated\tasks\task-2448.log"
RESULTS_CSV = os.path.abspath("runs/detect/runs/detect/cargo_yolo_exp_v2/results.csv")


def parse_latest_training_state():
    # Ensure utf-8 stdout
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 88)
    print("CARGO VISION - LIVE YOLOv8n TRAINING PROGRESS MONITOR (EXP V2)")
    print("=" * 88)

    # 1. Inspect results.csv for latest completed epochs
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "r", encoding="utf-8") as cf:
            clines = [c.strip() for c in cf.readlines() if c.strip()]
        if len(clines) > 1:
            total_completed = len(clines) - 1
            print(f"[STATUS] Total Completed Epochs Recorded: {total_completed} / 50")
            print("\n--- RECENT 5 COMPLETED EPOCHS ---")
            print(f"{'Epoch':<6} | {'Box Loss':<10} | {'Cls Loss':<10} | {'DFL Loss':<10} | {'Precision':<10} | {'Recall':<10} | {'mAP@50':<10} | {'mAP@50-95':<10}")
            print("-" * 88)
            for row in clines[-5:]:
                parts = [p.strip() for p in row.split(",")]
                if len(parts) >= 9 and parts[0] != "epoch":
                    ep = parts[0]
                    box_l = parts[2][:7]
                    cls_l = parts[3][:7]
                    dfl_l = parts[4][:7]
                    p = f"{float(parts[5])*100:.2f}%"
                    r = f"{float(parts[6])*100:.2f}%"
                    m50 = f"{float(parts[7])*100:.2f}%"
                    m95 = f"{float(parts[8])*100:.2f}%"
                    print(f"{ep:<6} | {box_l:<10} | {cls_l:<10} | {dfl_l:<10} | {p:<10} | {r:<10} | {m50:<10} | {m95:<10}")

    # 2. Inspect active log stream for in-progress batch
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Split on both \r and \n to handle dynamic terminal progress bars
        tokens = [t.strip() for t in re.split(r"[\r\n]+", content) if t.strip()]
        progress_tokens = [t for t in tokens if re.search(r"\b\d+/50\b", t)]

        if progress_tokens:
            latest_bar = progress_tokens[-1]
            safe_bar = latest_bar.encode("ascii", "replace").decode("ascii")
            print(f"\n--- CURRENT ACTIVE BATCH ---")
            print(f"  > {safe_bar}")

    print("\n" + "=" * 88)


if __name__ == "__main__":
    parse_latest_training_state()
