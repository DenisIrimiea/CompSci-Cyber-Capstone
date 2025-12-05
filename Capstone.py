#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import re
import json

# ---------------------------------------------------------
# Run helper script and capture stdout
# ---------------------------------------------------------

def run_script(script, *args):
    cmd = ["python", script] + list(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=60
        )
        return True, result.stdout
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------
# Extract boolean flag from console text
# ---------------------------------------------------------

def extract_flag(pattern, text):
    m = re.search(pattern, text)
    if not m:
        return None
    return m.group(1).strip().lower() == "true"


# ---------------------------------------------------------
# Extract background gradient from JSON output
# ---------------------------------------------------------

def parse_background_json(text):
    """
    Background gradient prints ONLY JSON. Example:

    {
      "background_gradient": false,
      "is_background_gradient": false,
      ...
    }
    """

    try:
        data = json.loads(text)
        if "background_gradient" in data:
            return bool(data["background_gradient"])
        if "is_background_gradient" in data:
            return bool(data["is_background_gradient"])
        return None
    except Exception:
        return None


# ---------------------------------------------------------
# Process a single FITS file
# ---------------------------------------------------------

def process_one_file(fits_file: Path):
    fits_str = str(fits_file)
    name = fits_file.name

    print(f"\nProcessing {name}")

    # ---------- Extractor ----------
    ok, out_extract = run_script("1Dextraction.py", "--fits", fits_str)
    low_snr = extract_flag(r"LOW_SNR:\s*(True|False)", out_extract)
    overexp = extract_flag(r"FIRST-ORDER OVEREXPOSED:\s*(True|False)", out_extract)

    # ---------- Star Streak ----------
    ok, out_streak = run_script("star_streak.py", "--fits", fits_str)
    star_streak = extract_flag(r"STAR_STREAK:\s*(True|False)", out_streak)

    # ---------- Partial first order ----------
    ok, out_partial = run_script("partial_first_order.py", "--fits", fits_str)
    partial_first = extract_flag(r"PARTIAL_FIRST_ORDER:\s*(True|False)", out_partial)

    # ---------- Background gradient (JSON!) ----------
    ok, out_bg = run_script("Background gradient.py", "--fits", fits_str)
    bg_grad = parse_background_json(out_bg)

    # ---------- Print flags ----------
    print("  LOW_SNR:              ", low_snr)
    print("  STAR_STREAK:          ", star_streak)
    print("  PARTIAL_FIRST_ORDER:  ", partial_first)
    print("  BACKGROUND_GRADIENT:  ", bg_grad)
    print("  OVEREXPOSURE:         ", overexp)

    # Collect flags dictionary
    flags = {
        "LOW_SNR": low_snr,
        "STAR_STREAK": star_streak,
        "PARTIAL_FIRST_ORDER": partial_first,
        "BACKGROUND_GRADIENT": bg_grad,
        "OVEREXPOSURE": overexp
    }

    # ---------- Classification ----------
    if (low_snr is True) or (star_streak is True):
        return "RED", name, flags

    if (partial_first is True) or (bg_grad is True) or (overexp is True):
        return "YELLOW", name, flags

    return "GREEN", name, flags


# ---------------------------------------------------------
# Batch mode
# ---------------------------------------------------------

def run_batch():
    fits_files = sorted(Path(".").glob("*.fits")) + sorted(Path(".").glob("*.fit"))

    green_data, yellow_data, red_data = [], [], []

    for f in fits_files:
        level, name, flags = process_one_file(f)

        block = (
            f"{name}\n"
            f"  LOW_SNR: {flags['LOW_SNR']}\n"
            f"  STAR_STREAK: {flags['STAR_STREAK']}\n"
            f"  PARTIAL_FIRST_ORDER: {flags['PARTIAL_FIRST_ORDER']}\n"
            f"  BACKGROUND_GRADIENT: {flags['BACKGROUND_GRADIENT']}\n"
            f"  OVEREXPOSURE: {flags['OVEREXPOSURE']}\n"
        )

        if level == "GREEN":
            green_data.append(block)
        elif level == "YELLOW":
            yellow_data.append(block)
        else:
            red_data.append(block)

    Path("green.txt").write_text("\n".join(green_data))
    Path("yellow.txt").write_text("\n".join(yellow_data))
    Path("red.txt").write_text("\n".join(red_data))

    print("\n=== SUMMARY ===")
    print("GREEN :", len(green_data))
    print("YELLOW:", len(yellow_data))
    print("RED   :", len(red_data))


# ---------------------------------------------------------
# Single file mode
# ---------------------------------------------------------

def run_single(f):
    level, name, flags = process_one_file(Path(f))
    print(f"\nRESULT FOR {name}: {level}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits")
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()

    if args.fits:
        run_single(args.fits)
        return

    if args.batch:
        run_batch()
        return

    print("Use --fits file.fit or --batch")


if __name__ == "__main__":
    main()
