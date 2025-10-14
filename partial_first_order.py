"""
partialFirstOrder.py
============================================================
Name: Brandon, Denis, Palmer, 06 Oct 2025
Section: CS453
Project: Capstone – Partial First-Order Detection (Slitless Spectra)
Purpose:
  CLI tool that loads a reduced 2-D FITS image, builds a per-column
  aperture-vs-background SNR mask for the first order, computes
  coverage, longest internal gap, and edge truncation metrics, and
  returns a verdict plus diagnostics in JSON.
============================================================
"""

from astropy.io import fits
import numpy as np
import argparse
import json


def extract_1d_from_2d(
    img2d: np.ndarray,
    row_center: int | None = None,
    half_ap: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    @brief Collapse a small horizontal aperture around the trace into a 1-D spectrum.
    @param img2d 2-D reduced image (rows, cols).
    @param row_center Optional known center row of the spectral trace.
    @param half_ap Half-height of the extraction aperture in pixels.
    @return (x, y) where x is pixel index (0..W-1) and y is extracted 1-D counts.
    """
    # Dimensions of the 2-D image (rows = height, cols = width).
    h, w = img2d.shape

    # If the trace center is unknown, pick a robust row: argmax of median-per-row.
    if row_center is None:
        row_summaries = np.median(img2d, axis=1)
        row_center = int(np.argmax(row_summaries))
        # Keep the aperture fully inside image bounds.
        row_center = int(np.clip(row_center, half_ap, h - half_ap - 1))

    # Sum a band of rows around the trace center to boost SNR.
    y = img2d[row_center - half_ap: row_center + half_ap + 1, :].sum(axis=0)

    # Use pixel index as x-axis when no wavelength solution is provided.
    x = np.arange(y.size)
    return x, y


def detect_partial_first_order_2d(
    img2d: np.ndarray,
    row_center: int | None = None,
    half_ap: int = 3,
    half_bg: int = 6,
    bg_gap: int = 1,
    snr_thresh: float = 3.0,
    min_coverage: float = 0.90,
    max_gap_px: int = 50,
    edge_trunc_px: int = 20
) -> tuple[bool, dict]:
    """
    @brief Assess whether the first-order spectrum is only partially present.
    @param img2d 2-D reduced image (rows, cols).
    @param row_center Optional known center row of the spectral trace.
    @param half_ap Half-height of the extraction aperture in pixels.
    @param half_bg Half-height of each background sideband (above/below).
    @param bg_gap Gap pixels between aperture edge and background sideband.
    @param snr_thresh Per-column SNR threshold for "detected".
    @param min_coverage Minimum fraction of columns that must be detected.
    @param max_gap_px Maximum allowed length of any internal undetected run.
    @param edge_trunc_px Undetected run length from an edge that implies truncation.
    @return (is_partial, info) verdict and diagnostics (coverage, gaps, edges, mask, params).
    """
    # Dimensions and safe placement of aperture.
    h, w = img2d.shape
    if row_center is None:
        row_summaries = np.median(img2d, axis=1)
        row_center = int(np.argmax(row_summaries))
    rc = int(np.clip(int(row_center), half_ap, h - half_ap - 1))

    # Aperture band and sidebands for background.
    ap_top, ap_bot = rc - half_ap, rc + half_ap + 1
    ap_band = img2d[ap_top:ap_bot, :]
    ap_sum = ap_band.sum(axis=0).astype(float)
    ap_npix = ap_band.shape[0]

    bg1_top, bg1_bot = max(0, ap_top - bg_gap - half_bg), max(0, ap_top - bg_gap)
    bg2_top, bg2_bot = min(h, ap_bot + bg_gap), min(h, ap_bot + bg_gap + half_bg)

    bgs = []
    if bg1_bot > bg1_top:
        bgs.append(img2d[bg1_top:bg1_bot, :])
    if bg2_bot > bg2_top:
        bgs.append(img2d[bg2_top:bg2_bot, :])

    if bgs:
        bg_stack = np.vstack(bgs)
        bg_mean = bg_stack.mean(axis=0)
        bg_var = bg_stack.var(axis=0) + 1e-9
    else:
        # Fallback if sidebands unavailable: use robust global background stats.
        bg_mean = np.median(img2d, axis=0)
        bg_var = np.var(img2d, axis=0) + 1e-9

    # Net counts and approximate background-limited SNR per column.
    net = ap_sum - bg_mean * ap_npix
    snr = net / np.sqrt(ap_npix * bg_var + 1e-9)

    # Columns where first-order signal is confidently present.
    detected = (snr >= snr_thresh)

    # Coverage across dispersion axis.
    coverage = float(detected.mean())

    # Edge runs of undetected columns (left and right).
    idx_false = np.where(~detected)[0]
    edge_left = edge_right = 0
    if idx_false.size:
        i = 0
        while i < idx_false.size and idx_false[i] == i:
            i += 1
        edge_left = i
        j = 0
        while j < idx_false.size and idx_false[-(j + 1)] == (w - 1 - j):
            j += 1
        edge_right = j

    # Longest internal undetected run (exclude pure edge runs).
    max_gap = 0
    if w > 0:
        gaps, start = [], None
        for c in range(w):
            if not detected[c]:
                if start is None:
                    start = c
            else:
                if start is not None:
                    gaps.append((start, c - 1))
                    start = None
        if start is not None:
            gaps.append((start, w - 1))
        internal = [(a, b) for (a, b) in gaps if a > 0 and b < (w - 1)]
        if internal:
            max_gap = int(max(b - a + 1 for (a, b) in internal))

    # Decision: low coverage OR long internal gaps OR pronounced edge truncation.
    is_partial = (
        (coverage < min_coverage) or
        (max_gap >= max_gap_px) or
        (edge_left >= edge_trunc_px) or
        (edge_right >= edge_trunc_px)
    )

    # Package diagnostics for audit and tuning.
    info = {
        "coverage": coverage,
        "snr_thresh": snr_thresh,
        "max_gap_px": max_gap,
        "edge_left_px": edge_left,
        "edge_right_px": edge_right,
        "min_coverage": min_coverage,
        "edge_trunc_px": edge_trunc_px,
        "half_ap": half_ap,
        "half_bg": half_bg,
        "bg_gap": bg_gap,
        "mask_detected": detected.tolist()
    }
    return bool(is_partial), info


def detect_partial_first_order_fits(
    fits_path: str,
    hdu_index: int = 0,
    row_center: int | None = None,
    half_ap: int = 3,
    half_bg: int = 6,
    bg_gap: int = 1,
    snr_thresh: float = 3.0,
    min_coverage: float = 0.90,
    max_gap_px: int = 50,
    edge_trunc_px: int = 20
) -> tuple[bool, dict]:
    """
    @brief Open a 2-D reduced FITS image and run the partial-first-order detector.
    @param fits_path Path to the FITS file containing a reduced 2-D image.
    @param hdu_index HDU index to read (default 0).
    @param row_center Optional known center row of the spectral trace.
    @param half_ap Half-height of the extraction aperture in pixels.
    @param half_bg Half-height of background sidebands (pixels).
    @param bg_gap Gap pixels between aperture and background sidebands.
    @param snr_thresh Per-column SNR threshold for detection.
    @param min_coverage Minimum fraction of detected columns.
    @param max_gap_px Max internal undetected run allowed (pixels).
    @param edge_trunc_px Edge run length implying truncation (pixels).
    @return (is_partial, info) verdict and diagnostics.
    """
    # Open FITS with memory mapping for efficiency on large files.
    with fits.open(fits_path, memmap=True) as hdul:
        data = np.asarray(hdul[hdu_index].data)

        # Validate that the selected HDU is a 2-D image (Case A workflow).
        if data.ndim != 2:
            raise ValueError("Expected a 2-D reduced image for Case A.")

    # Run detector directly on the 2-D image (uses same row-center heuristic).
    return detect_partial_first_order_2d(
        data,
        row_center=row_center,
        half_ap=half_ap,
        half_bg=half_bg,
        bg_gap=bg_gap,
        snr_thresh=snr_thresh,
        min_coverage=min_coverage,
        max_gap_px=max_gap_px,
        edge_trunc_px=edge_trunc_px
    )


if __name__ == "__main__":
    # CLI: parse arguments and run the detector for the provided FITS file.
    parser = argparse.ArgumentParser(
        description="Partial first-order detector (2-D FITS -> SNR mask across columns)."
    )
    parser.add_argument("--fits", required=True, help="Path to reduced 2-D FITS image")
    parser.add_argument("--hdu", type=int, default=0, help="HDU index (default 0)")
    parser.add_argument(
        "--half-ap", type=int, default=3, help="Half aperture height in pixels"
    )
    parser.add_argument(
        "--row-center", type=int, default=None,
        help="Row index of trace center (optional)"
    )
    parser.add_argument("--half-bg", type=int, default=6,
                        help="Background sideband half-height (pixels)")
    parser.add_argument("--bg-gap", type=int, default=1,
                        help="Gap between aperture and background (pixels)")
    parser.add_argument("--snr-thresh", type=float, default=3.0,
                        help="Per-column SNR threshold for detection")
    parser.add_argument("--min-coverage", type=float, default=0.90,
                        help="Minimum detected fraction to avoid partial flag")
    parser.add_argument("--max-gap-px", type=int, default=50,
                        help="Max length of any internal undetected run (pixels)")
    parser.add_argument("--edge-trunc-px", type=int, default=20,
                        help="Undetected run from an edge implying truncation (pixels)")

    args = parser.parse_args()

    is_partial, info = detect_partial_first_order_fits(
        args.fits,
        hdu_index=args.hdu,
        row_center=args.row_center,
        half_ap=args.half_ap,
        half_bg=args.half_bg,
        bg_gap=args.bg_gap,
        snr_thresh=args.snr_thresh,
        min_coverage=args.min_coverage,
        max_gap_px=args.max_gap_px,
        edge_trunc_px=args.edge_trunc_px
    )

    # Emit structured JSON for logging/automation; exit code remains 0.
    print(json.dumps({"partial_first_order": bool(is_partial), **info}, indent=2))