#!/usr/bin/env python3
"""
Improved star streak detector with console reporting.
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
# STAR STREAK DETECTION
# =========================================================

def detect_star_streak(s, profile):
    window = min(101, len(profile) - (len(profile) % 2 == 0))
    smoothed = savgol_filter(profile, window, 3)

    median_back = np.median(smoothed)

    peaks, props = find_peaks(
        smoothed,
        height=median_back * 1.2,
        prominence=median_back * 1.5,
        width=1,
        distance=int(0.02 * len(s))
    )

    heights     = props["peak_heights"]
    prominences = props["prominences"]
    widths      = props["widths"]

    if len(peaks) < 2:
        return False, peaks, smoothed, props, [False] * len(peaks)

    sorted_idx  = np.argsort(heights)[::-1]
    zeroth_idx  = sorted_idx[0]
    first_idx   = sorted_idx[1]

    zeroth_pos  = peaks[zeroth_idx]
    first_pos   = peaks[first_idx]

    valid_mask = []
    has_streak = False

    for pk, prom, w, h in zip(peaks, prominences, widths, heights):

        if pk == zeroth_pos or pk == first_pos:
            valid_mask.append(True)
            continue

        after_first      = pk > first_pos
        strong_enough    = prom > 0.7 * prominences[first_idx]
        narrow_enough    = w < 250
        above_background = h > median_back * 1.2

        if after_first and (strong_enough or narrow_enough or above_background):
            valid_mask.append(True)
            has_streak = True
        else:
            valid_mask.append(False)

    return has_streak, peaks, smoothed, props, valid_mask


# =========================================================
# PROCESS FILE
# =========================================================

def process_single_file(fits_path: Path, extraction_script="1Dextraction.py"):

    prefix = fits_path.stem
    fits_file = str(fits_path)

    print(f"\nProcessing {fits_file}")

    subprocess.run([
        "python", extraction_script,
        "--fits", fits_file,
        "--prefix", prefix
    ], check=True)

    parent  = fits_path.parent
    txt_path = parent / f"{prefix}_line_1d.txt"
    png_path = parent / f"{prefix}_line_spectrum.png"

    arr = np.loadtxt(txt_path, skiprows=1)
    s, profile = arr[:,0], arr[:,1]

    img_png = imread(png_path)

    streak, peaks, smoothed, props, valid_mask = detect_star_streak(s, profile)

    out_png = parent / f"{prefix}_streak_detection.png"

    # ------------------------- Plot -------------------------
    plt.figure(figsize=(14, 10))
    plt.suptitle(f"Streak Detection: {'YES' if streak else 'NO'}", fontsize=18)

    plt.subplot(2, 1, 1)
    plt.imshow(img_png)
    plt.title(f"Extraction: {prefix}")
    plt.axis("off")

    plt.subplot(2, 1, 2)
    plt.plot(s, profile, alpha=0.4)
    plt.plot(s, smoothed, linewidth=2)

    for i, pk in enumerate(peaks):
        plt.axvline(s[pk], color="green" if valid_mask[i] else "red", linestyle="--")

    plt.xlabel("Distance (pixels)")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # ---------------------- Console Logging ----------------------

    print("\n===================== STAR STREAK REPORT =====================")
    print(f"FILE:          {fits_path.name}")
    print(f"STAR_STREAK:   {streak}")
    print(f"PEAK_POSITIONS: {peaks.tolist()}")
    print(f"VALID_MASK:     {valid_mask}")

    if len(peaks) >= 2:
        sorted_idx  = np.argsort(props['peak_heights'])[::-1]
        zeroth_pos  = peaks[sorted_idx[0]]
        first_pos   = peaks[sorted_idx[1]]
        print(f"ZEROTH_ORDER_POSITION: {zeroth_pos}")
        print(f"FIRST_ORDER_POSITION:  {first_pos}")
    else:
        print("Not enough peaks to determine zeroth/first order.")

    print(f"OUTPUT_IMAGE:  {out_png}")
    print("==============================================================\n")

    return prefix, streak


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", required=False)
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()

    if args.batch:
        fits_files = sorted(Path(".").glob("*.fits")) + sorted(Path(".").glob("*.fit"))
        results = []
        for f in fits_files:
            prefix, streak = process_single_file(f)
            results.append((prefix, streak))

        with open("streak_summary.csv", "w", newline="") as f:
            csv.writer(f).writerows([["Prefix","StreakDetected"]] + results)

        print("\nBatch complete! Saved streak_summary.csv\n")
        return

    if args.fits:
        process_single_file(Path(args.fits))
        return

    print("Provide --fits <file> or --batch")


if __name__ == "__main__":
    main()

