#!/usr/bin/env python3
"""
lowSNR.py
============================================================
Low-SNR detector for slitless spectroscopy (FITS)

Pipeline (horizontal dispersion assumed):
  1) Auto-detect (or accept) aperture row.
  2) Extract aperture sum vs. column and two background sidebands.
  3) Estimate noise per column from background (robust MAD), scale to
     aperture-sum noise (sqrt of #aperture pixels).
  4) Compute net spectrum: ap_sum - bg_track; smooth with Savitzky–Golay.
  5) Build metrics:
       - Global dynamic-range SNR (P95-P5)/ (6 * median sigma_ap)
       - Per-column SNR: net / sigma_ap -> take 95th percentile
       - Coverage: fraction of columns with SNR >= snr_col_thresh
       - HF/LF ratio: std of residuals after heavy smoothing vs. continuum dyn
  6) Verdict = LOW SNR if multiple conservative gates trip (tunable).
  7) Output JSON with metrics, gates, and a score (0..4).

Assumptions:
- Noise dominated by background shot/read noise captured in sidebands.
- Background-track variance approximates aperture-sum noise scaled by
  sqrt(N_aperture_pixels). This is conservative and instrument-agnostic.

Dependencies: numpy, astropy, scipy
============================================================
"""

import argparse
import json
import sys
from typing import Tuple, Dict, Optional

import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr


# ------------------------------
# FITS I/O
# ------------------------------
def load_fits_2d(path: str) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        for h in hdul:
            if getattr(h, "data", None) is not None and isinstance(h.data, np.ndarray) and h.data.ndim == 2:
                return np.asarray(h.data, dtype=float)
    raise ValueError("No 2D image found in FITS.")


# ------------------------------
# Auto-detect aperture row
# ------------------------------
def autodetect_ap_row(
    img: np.ndarray,
    method: str = "median",
    col_trim_frac: float = 0.05,
    smooth_px: int = 31,
    refine_half: int = 6,
) -> int:
    H, W = img.shape
    trim = int(max(0, min(W // 2, round(col_trim_frac * W))))
    xlo, xhi = trim, W - trim
    if xhi - xlo < 8:
        xlo, xhi = 0, W

    core = img[:, xlo:xhi]
    if method == "mean":
        row_profile = np.nanmean(core, axis=1)
    else:
        row_profile = np.nanmedian(core, axis=1)

    smooth_px = max(3, int(smooth_px))
    row_sm = uniform_filter1d(row_profile, size=smooth_px, mode="nearest")

    y_peak = int(np.argmax(row_sm))
    lo = max(0, y_peak - refine_half)
    hi = min(H, y_peak + refine_half + 1)
    y_idx = np.arange(lo, hi, dtype=float)
    wts = np.maximum(row_sm[lo:hi] - np.min(row_sm[lo:hi]), 0.0) + 1e-12
    y_com = float(np.sum(y_idx * wts) / np.sum(wts))
    return int(np.clip(round(y_com), 0, H - 1))


# ------------------------------
# Extraction and background bands
# ------------------------------
def extract_aperture_and_background(
    img: np.ndarray,
    ap_row: int,
    half_ap: int = 4,
    half_bg: int = 8,
    bg_gap: int = 2,
    edge_trunc_px: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      x:              1D array of column indices
      ap_sum:         aperture-summed counts vs x
      bg_track:       per-column background (median across both bands)
      net:            ap_sum - bg_track
      sigma_ap_col:   per-column noise estimate for the aperture sum
    """
    H, W = img.shape
    y0 = int(ap_row)

    ap_y_lo = max(0, y0 - half_ap)
    ap_y_hi = min(H, y0 + half_ap + 1)
    ap_h = ap_y_hi - ap_y_lo
    if ap_h <= 0:
        raise ValueError("Aperture height is zero; check ap_row/half_ap.")

    # background bands (top and bottom)
    bg1_lo = max(0, ap_y_hi + bg_gap)
    bg1_hi = min(H, bg1_lo + 2 * half_bg)
    bg2_hi = max(0, ap_y_lo - bg_gap)
    bg2_lo = max(0, bg2_hi - 2 * half_bg)

    if (bg1_hi - bg1_lo) <= 0 or (bg2_hi - bg2_lo) <= 0:
        raise ValueError("Background bands out of bounds; adjust geometry.")

    # Aperture sums
    ap_strip = img[ap_y_lo:ap_y_hi, :]    # shape (ap_h, W)
    ap_sum = np.sum(ap_strip, axis=0)

    # Background stack and per-column median (track) + per-column noise
    bg_band_top = img[bg1_lo:bg1_hi, :]   # shape (2*half_bg, W)
    bg_band_bot = img[bg2_lo:bg2_hi, :]
    bg_stack = np.concatenate([bg_band_top, bg_band_bot], axis=0)  # (Nb, W)
    bg_track = np.median(bg_stack, axis=0)

    # Robust per-column noise on background pixels (MAD -> sigma)
    # MAD across rows gives an estimate of per-pixel noise at each column.
    # Aperture-sum noise scales ~ sqrt(ap_h).
    mad = 1.4826 * np.median(np.abs(bg_stack - np.median(bg_stack, axis=0, keepdims=True)), axis=0)
    sigma_ap_col = np.sqrt(max(ap_h, 1)) * mad  # conservative

    x = np.arange(W, dtype=int)
    if edge_trunc_px > 0:
        sl = slice(edge_trunc_px, W - edge_trunc_px)
        x = x[sl]
        ap_sum = ap_sum[sl]
        bg_track = bg_track[sl]
        sigma_ap_col = sigma_ap_col[sl]

    net = ap_sum - bg_track
    return x, ap_sum, bg_track, net, sigma_ap_col


# ------------------------------
# λ handling and smoothing
# ------------------------------
def to_lambda(x: np.ndarray, wl0: Optional[float], dlambda: Optional[float]) -> np.ndarray:
    if wl0 is not None and dlambda is not None:
        return wl0 + x.astype(float) * float(dlambda)
    return x.astype(float)


def sg_smooth(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    poly = max(1, int(poly))
    poly = min(poly, window - 1)
    return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")


# ------------------------------
# Low-SNR metrics
# ------------------------------
def compute_low_snr_metrics(
    lam: np.ndarray,
    net: np.ndarray,
    sigma_ap_col: np.ndarray,
    sg_window: int = 21,
    sg_poly: int = 3,
    heavy_sg_window: int = 101,
) -> Dict:
    """
    Build several complementary SNR indicators from the 1-D net spectrum and
    per-column aperture noise estimate.
    """
    # Lightweight smoothing (like your star-streak)
    net_s = sg_smooth(net, sg_window, sg_poly)

    # Global dynamic range of the smoothed net
    dyn = float(np.percentile(net_s, 95) - np.percentile(net_s, 5))

    # Median per-column sigma (scalar)
    sigma_med = float(np.median(sigma_ap_col))

    # Global dynamic-range SNR (approx P95-P5 ~ 6σ if Gaussian)
    snr_global = float(dyn / (6.0 * (sigma_med if sigma_med > 0 else 1e-12)))

    # Per-column SNR; ignore columns with zero/neg sigma
    denom = np.where(sigma_ap_col > 0, sigma_ap_col, np.nan)
    snr_col = net / denom
    snr_col_s = net_s / denom

    # Conservative positives: only where net_s > 0 (avoid negative-baseline artifacts)
    pos = net_s > 0
    snr_col_pos = snr_col_s[pos] if np.any(pos) else np.array([np.nan])

    snr_p95 = float(np.nanpercentile(snr_col_pos, 95)) if np.isfinite(snr_col_pos).any() else float("nan")
    snr_med = float(np.nanmedian(snr_col_pos)) if np.isfinite(snr_col_pos).any() else float("nan")

    # Coverage: fraction of columns with SNR above modest threshold (e.g., 1.5)
    snr_col_thresh = 1.5
    coverage = float(np.nanmean((snr_col_s >= snr_col_thresh).astype(float)))

    # High-frequency vs low-frequency content
    # Heavy smoothing to get LF continuum, residual is HF
    net_heavy = sg_smooth(net, heavy_sg_window, sg_poly)
    resid = net - net_heavy
    hf_std = float(np.std(resid))
    lf_dyn = float(np.percentile(net_heavy, 95) - np.percentile(net_heavy, 5))
    # Normalize HF energy by LF scale + a small floor to avoid div-by-zero
    hf_lf_ratio = float(hf_std / (lf_dyn / 6.0 + 1e-9))

    # Smoothness (optional diagnostic): Spearman correlation between λ and heavy continuum
    rho, _ = spearmanr(lam, net_heavy)

    return {
        "dyn_5to95": dyn,
        "sigma_med": sigma_med,
        "snr_global": snr_global,
        "snr_p95": snr_p95,
        "snr_med_pos": snr_med,
        "coverage_snr_ge_1p5": coverage,
        "hf_lf_ratio": hf_lf_ratio,
        "spearman_lam_continuum": float(rho) if np.isfinite(rho) else 0.0,
        "snr_col_thresh": snr_col_thresh,
    }


def low_snr_verdict(
    m: Dict,
    snr_global_min: float = 1.0,
    snr_p95_min: float = 3.0,
    coverage_min: float = 0.20,
    hf_lf_max: float = 1.0,
) -> Dict:
    """
    Multi-gate decision; returns boolean and which gates fired.
    Gates are intentionally conservative and tunable:

      A) snr_global < snr_global_min
      B) snr_p95    < snr_p95_min
      C) coverage   < coverage_min
      D) hf_lf_ratio > hf_lf_max      (noise-like)
    """
    gates = {
        "A_snr_global_low": (m["snr_global"] < snr_global_min),
        "B_snr_p95_low": (np.isfinite(m["snr_p95"]) and (m["snr_p95"] < snr_p95_min)),
        "C_coverage_low": (m["coverage_snr_ge_1p5"] < coverage_min),
        "D_hf_lf_high": (m["hf_lf_ratio"] > hf_lf_max),
    }

    score = int(sum(bool(v) for v in gates.values()))
    # Require at least 2 independent gates to avoid false trips
    verdict = score >= 2

    return {"low_snr": bool(verdict), "low_snr_score": int(score), "gates": gates}


# ------------------------------
# CLI
# ------------------------------
def main():
    p = argparse.ArgumentParser(description="Low-SNR detector (FITS -> 1D net -> SNR metrics -> verdict)")
    p.add_argument("--fits", required=True, help="Path to 2-D FITS with dispersion along X")

    # Aperture geometry
    p.add_argument("--ap-row", type=int, default=None, help="Aperture center row (Y index). If omitted, auto-detect.")
    p.add_argument("--half-ap", type=int, default=4, help="Half-aperture height in pixels (aperture thickness = 2*half_ap+1)")
    p.add_argument("--half-bg", type=int, default=8, help="Half-background band half-height (each band thickness = 2*half_bg)")
    p.add_argument("--bg-gap", type=int, default=2, help="Vertical pixel gap between aperture and each background band")
    p.add_argument("--edge-trunc-px", type=int, default=0, help="Discard this many pixels from BOTH left and right edges")

    # Auto-detection controls
    p.add_argument("--ap-auto", action="store_true", help="Force auto-detection even if --ap-row provided")
    p.add_argument("--ap-method", choices=["median", "mean"], default="median", help="Row profile collapse method for auto-detect")
    p.add_argument("--ap-col-trim-frac", type=float, default=0.05, help="Trim this fraction of columns on BOTH sides before row profile")
    p.add_argument("--ap-smooth-px", type=int, default=31, help="Row-profile smoothing width (rows)")
    p.add_argument("--ap-refine-half", type=int, default=6, help="Half window for center-of-mass refinement")

    # Wavelength calibration (optional, for reporting)
    p.add_argument("--wl0", type=float, default=None, help="λ at x=0 (nm)")
    p.add_argument("--dlambda", type=float, default=None, help="Dispersion nm per pixel")

    # Smoothing
    p.add_argument("--sg-window", type=int, default=21, help="SavGol window length (odd)")
    p.add_argument("--sg-poly", type=int, default=3, help="SavGol polynomial order")
    p.add_argument("--heavy-sg-window", type=int, default=101, help="Heavier SavGol window for HF/LF split (odd)")

    # Decision thresholds (tunable)
    p.add_argument("--snr-global-min", type=float, default=1.0, help="Gate A: min global dynamic-range SNR")
    p.add_argument("--snr-p95-min", type=float, default=3.0, help="Gate B: min 95th-percentile per-column SNR")
    p.add_argument("--coverage-min", type=float, default=0.20, help="Gate C: min fraction of columns with SNR>=1.5")
    p.add_argument("--hf-lf-max", type=float, default=1.0, help="Gate D: max HF/LF ratio (higher => noise-like)")

    args = p.parse_args()

    try:
        img = load_fits_2d(args.fits)

        # Auto-detect ap-row if needed
        ap_row_used = args.ap_row
        if ap_row_used is None or args.ap_auto:
            ap_row_used = autodetect_ap_row(
                img,
                method=args.ap_method,
                col_trim_frac=args.ap_col_trim_frac,
                smooth_px=args.ap_smooth_px,
                refine_half=args.ap_refine_half,
            )

        x, ap_sum, bg_track, net, sigma_ap_col = extract_aperture_and_background(
            img,
            ap_row=ap_row_used,
            half_ap=args.half_ap,
            half_bg=args.half_bg,
            bg_gap=args.bg_gap,
            edge_trunc_px=args.edge_trunc_px,
        )

        lam = to_lambda(x, wl0=args.wl0, dlambda=args.dlambda)

        metrics = compute_low_snr_metrics(
            lam=lam,
            net=net,
            sigma_ap_col=sigma_ap_col,
            sg_window=args.sg_window,
            sg_poly=args.sg_poly,
            heavy_sg_window=args.heavy_sg_window,
        )

        verdict = low_snr_verdict(
            metrics,
            snr_global_min=args.snr_global_min,
            snr_p95_min=args.snr_p95_min,
            coverage_min=args.coverage_min,
            hf_lf_max=args.hf_lf_max,
        )

        out = {
            "file": args.fits,
            "ap_row_input": args.ap_row,
            "ap_row_used": int(ap_row_used),
            "ap_auto": bool(args.ap_row is None or args.ap_auto),
            "half_ap": args.half_ap,
            "half_bg": args.half_bg,
            "bg_gap": args.bg_gap,
            "edge_trunc_px": args.edge_trunc_px,
            "wl0": args.wl0,
            "dlambda": args.dlambda,
            **metrics,
            **verdict,
        }

        print(json.dumps(out, indent=2))
    except Exception as e:
        err = {"error": str(e)}
        print(json.dumps(err, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
