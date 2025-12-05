#!/usr/bin/env python3
"""
partial_first_order.py

FINAL SIMPLE FIRST-ORDER CLASSIFIER:

PARTIAL FIRST ORDER = TRUE when:
    (tail_mean of last 1% of smoothed profile) > 150

Everything else is ignored.
"""

import argparse
import subprocess
from pathlib import Path
import csv
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
from matplotlib.image import imread
from astropy.io import fits


# =========================================================
# SIMPLE CLASSIFIER — ONLY uses tail_mean > 150
# =========================================================

def classify_first_order(s, profile):
    n = len(profile)
    if n < 20:
        return "NONE", False, {"reason": "too_short", "tail_mean": 0}, np.array([]), profile

    # Smooth safely
    window = min(101, n - (n % 2 == 0))
    if window < 7:
        window = 7
    if window % 2 == 0:
        window += 1

    smoothed = savgol_filter(profile, window_length=window, polyorder=3)

    # -------- TAIL MEAN (LAST 1%) --------
    tail_start = int(0.99 * n)
    tail_slice = smoothed[tail_start:]

    if len(tail_slice) == 0:
        tail_mean = float(np.median(smoothed))
    else:
        tail_mean = float(tail_slice.mean())

    info = {
        "tail_mean": tail_mean,
        "reason": "",
    }

    # RULE: tail_mean > 150 → PARTIAL
    if tail_mean > 150:
        info["reason"] = "tail_mean_gt_150"
        return "PARTIAL", True, info, np.array([]), smoothed

    # OTHERWISE FULL
    info["reason"] = "tail_mean_le_150"
    return "FULL", False, info, np.array([]), smoothed



# =========================================================
# PROCESS A SINGLE FILE
# =========================================================

def process_single_file(fits_path: Path, extraction_script="1Dextraction.py"):

    prefix = fits_path.stem
    abs_fits = fits_path.resolve()
    abs_ext  = Path(extraction_script).resolve()

    print(f"\nProcessing {abs_fits}")

    # Run extraction (required for 1D profile)
    subprocess.run([
        "python",
        str(abs_ext),
        "--fits", str(abs_fits),
        "--prefix", prefix
    ], check=True)

    parent = abs_fits.parent

    txt_path = parent / f"{prefix}_line_1d.txt"
    png_path = parent / f"{prefix}_line_spectrum.png"

    arr = np.loadtxt(txt_path, skiprows=1)
    s = arr[:, 0]
    profile = arr[:, 1]
    img_png = imread(png_path)

    # CLASSIFY USING NEW SIMPLE RULE
    first_class, is_partial, info, peaks, smoothed = classify_first_order(s, profile)

    out_png = parent / f"{prefix}_partial_first_order.png"

    # Plot
    plt.figure(figsize=(14, 10))
    plt.suptitle(
        f"FIRST ORDER: {first_class} (PARTIAL={is_partial})",
        fontsize=18, fontweight="bold"
    )

    plt.subplot(2, 1, 1)
    plt.imshow(img_png)
    plt.axis("off")
    plt.title(f"Extraction: {prefix}")

    plt.subplot(2, 1, 2)
    plt.plot(s, profile, alpha=0.5, label="Raw")
    plt.plot(s, smoothed, linewidth=2, label="Smoothed")
    plt.xlabel("Distance (pixels)")
    plt.ylabel("Counts")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # Output
    print("\n================ FIRST ORDER REPORT ================")
    print(f"FILE:                {abs_fits.name}")
    print(f"FIRST_ORDER_CLASS:   {first_class}")
    print(f"PARTIAL_FIRST_ORDER: {is_partial}")
    print(f"TAIL_MEAN:           {info['tail_mean']:.2f}")
    print(f"REASON:              {info['reason']}")
    print(f"OUTPUT_IMAGE:        {out_png}")
    print("====================================================\n")

    return prefix, first_class, is_partial



# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Simple partial first-order detector (tail rule only).")
    parser.add_argument("--fits")
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()

    if args.batch:
        fits_files = sorted(Path(".").glob("*.fit")) + sorted(Path(".").glob("*.fits"))
        results = []

        for f in fits_files:
            prefix, classification, is_partial = process_single_file(f)
            results.append((prefix, classification, is_partial))

        with open("partial_first_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Prefix", "FirstOrderClass", "PartialFirstOrder"])
            writer.writerows(results)

        print("\nBatch complete. Saved partial_first_summary.csv\n")
        return

    if args.fits:
        process_single_file(Path(args.fits))
        return

    print("ERROR: Provide --fits <file> or use --batch")


if __name__ == "__main__":
    main()
