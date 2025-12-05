#!/usr/bin/env python3
"""
Final star streak detector with:
 - Auto-prefix detection
 - Single-file mode
 - FULL BATCH MODE (process every FITS file in folder)

A streak is confirmed when:
 - There are ≥ 3 significant peaks
 - The 3rd peak occurs AFTER the first-order bump
 - Prominence > 2 × background median
 - Peak width < 150 px
"""

import argparse
import subprocess
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
from matplotlib.image import imread
import csv


# =========================================================
# STAR STREAK DETECTION LOGIC
# =========================================================

def detect_star_streak(s, profile):

    # Smooth profile
    window = min(101, len(profile) - (len(profile) % 2 == 0))
    smoothed = savgol_filter(profile, window, 3)

    median_back = np.median(smoothed)

    # Request peak_heights by specifying "height"
    peaks, props = find_peaks(
        smoothed,
        height=median_back * 2,
        prominence=median_back * 2,
        width=1,
        distance=int(0.03 * len(s))
    )

    prominences = props["prominences"]
    widths      = props["widths"]
    heights     = props["peak_heights"]

    if len(peaks) < 3:
        return False, peaks, smoothed, props, [False] * len(peaks)

    # Identify zeroth + first-order peaks
    sorted_idx = np.argsort(heights)[::-1]
    zeroth_idx = sorted_idx[0]
    first_idx  = sorted_idx[1]

    zeroth_pos = peaks[zeroth_idx]
    first_pos  = peaks[first_idx]

    valid_mask = []
    valid_count = 2  # zeroth + first

    # Evaluate remaining peaks
    for pk, p, w, h in zip(peaks, prominences, widths, heights):

        if pk == zeroth_pos or pk == first_pos:
            valid_mask.append(True)
            continue

        # strict streak rule
        if (
            pk > first_pos and
            p > median_back * 2 and
            w < 150
        ):
            valid_mask.append(True)
            valid_count += 1
        else:
            valid_mask.append(False)

    streak = valid_count >= 3
    return streak, peaks, smoothed, props, valid_mask


# =========================================================
# PROCESS A SINGLE FITS FILE
# =========================================================

def process_single_file(fits_path: Path, extraction_script="1Dextraction.py"):

    prefix = fits_path.stem
    fits_file = str(fits_path)

    print(f"\n🚀 Processing {fits_file} (prefix: {prefix})\n")

    # Run extraction
    subprocess.run([
        "python", extraction_script,
        "--fits", fits_file,
        "--prefix", prefix
    ], check=True)

    # Load outputs
    txt_path = Path(f"{prefix}_line_1d.txt")
    png_path = Path(f"{prefix}_line_spectrum.png")

    arr = np.loadtxt(txt_path, skiprows=1)
    s       = arr[:, 0]
    profile = arr[:, 1]

    img_png = imread(png_path)

    # Run streak detection
    streak, peaks, smoothed, props, valid_mask = detect_star_streak(s, profile)

    # Save diagnostic visualization
    out_png = f"{prefix}_streak_detection.png"

    plt.figure(figsize=(12, 9))

    # Extraction image
    plt.subplot(2, 1, 1)
    plt.imshow(img_png)
    plt.title(f"Extraction: {prefix}")
    plt.axis("off")

    # Peaks visualization
    plt.subplot(2, 1, 2)
    plt.plot(s, profile, alpha=0.4, label="Raw")
    plt.plot(s, smoothed, linewidth=2, label="Smoothed")

    for i, pk in enumerate(peaks):
        color = "green" if valid_mask[i] else "red"
        plt.axvline(s[pk], color=color, linestyle="--", alpha=0.7)

    plt.title(f"Streak: {'YES' if streak else 'NO'}")
    plt.xlabel("Distance (pixels)")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_png, dpi=200)

    print(f"📤 Saved: {out_png}")

    # Return result tuple
    return prefix, streak


# =========================================================
# MAIN — SINGLE MODE + BATCH MODE
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Star streak detector with batch mode.")
    parser.add_argument("--fits", required=False)
    parser.add_argument("--batch", action="store_true",
                        help="Process ALL .fit/.fits files in folder")
    args = parser.parse_args()

    # -------------------------
    # BATCH MODE
    # -------------------------
    if args.batch:
        print("\n🔥 BATCH MODE ENABLED — Processing all FITS files...\n")

        fits_files = sorted(list(Path(".").glob("*.fit")) +
                            list(Path(".").glob("*.fits")))

        if not fits_files:
            print("❌ No FITS files found in this directory.")
            return

        results = []

        for f in fits_files:
            prefix, streak = process_single_file(f)
            results.append((prefix, streak))

        # Save summary CSV
        with open("streak_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Prefix", "StreakDetected"])
            writer.writerows(results)

        print("\n✅ Batch processing complete.")
        print("📄 Summary saved to streak_summary.csv\n")
        return

    # -------------------------
    # SINGLE-FILE MODE
    # -------------------------
    if args.fits:
        fits_path = Path(args.fits)
        process_single_file(fits_path)
        return

    print("❌ ERROR: Provide --fits <file> or use --batch")


if __name__ == "__main__":
    main()
