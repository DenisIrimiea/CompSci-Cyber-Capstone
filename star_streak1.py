# star_streak_cli.py

from astropy.io import fits

import numpy as np

from scipy.signal import savgol_filter, find_peaks

import argparse, json
 
def mad(x):

    med = np.median(x)

    return 1.4826 * np.median(np.abs(x - med))
 
def extract_1d_from_2d(img2d, row_center=None, half_ap=3):

    h, w = img2d.shape

    if row_center is None:

        row_sums = np.median(img2d, axis=1)

        row_center = int(np.argmax(row_sums))

        row_center = np.clip(row_center, half_ap, h - half_ap - 1)

    y = img2d[row_center - half_ap: row_center + half_ap + 1, :].sum(axis=0)

    x = np.arange(y.size)

    return x, y
 
def detect_star_streak_1d(x, y, sg_window=31, sg_poly=3, z_prom=6.0, w_max=4,

                          run_window=200, n_min=2, s_min=8.0, ignore_edge=10,

                          check_negative=True):

    n = len(y)

    if n < max(sg_window + 2, 50):

        raise ValueError("1-D array too short for stable detection.")

    if sg_window % 2 == 0: sg_window += 1

    sg_window = min(sg_window, n - 1 - (n % 2 == 0))

    y_hat = savgol_filter(y, window_length=sg_window, polyorder=sg_poly, mode='interp')

    r = y - y_hat

    sigma = mad(r) or (np.std(r) + 1e-9)

    z = r / sigma
 
    def _find(zarr):

        peaks, props = find_peaks(zarr, prominence=z_prom, width=(1, w_max))

        keep = (peaks >= ignore_edge) & (peaks < (n - ignore_edge))

        return peaks[keep], {k: v[keep] for k, v in props.items()}
 
    ppos, prpos = _find(z)

    pneg, prneg = _find(-z) if check_negative else (np.array([]), {})

    peaks = np.concatenate([ppos, pneg]).astype(int)

    prom  = np.concatenate([prpos.get('prominences', np.array([])),

                            prneg.get('prominences', np.array([]))]) if peaks.size else np.array([])

    widths = np.concatenate([prpos.get('widths', np.array([])),

                             prneg.get('widths', np.array([]))]) if peaks.size else np.array([])
 
    has_run = False

    if peaks.size:

        order = np.argsort(peaks)

        peaks, prom, widths = peaks[order], prom[order], widths[order]

        i0 = 0

        for i1 in range(peaks.size):

            while peaks[i1] - peaks[i0] > run_window:

                i0 += 1

            if (i1 - i0 + 1) >= n_min:

                has_run = True

                break
 
    score = float(np.sum(np.maximum(0.0, prom - (z_prom - 0.5))))

    has_streak = (peaks.size >= 1 and score >= s_min) or has_run
 
    return bool(has_streak), {

        "peaks_idx": peaks.tolist(),

        "prominences": prom.tolist(),

        "widths": widths.tolist(),

        "score": score,

        "sigma": float(sigma),

        "params": {

            "sg_window": sg_window, "sg_poly": sg_poly,

            "z_prom": z_prom, "w_max": w_max,

            "run_window": run_window, "n_min": n_min,

            "s_min": s_min, "ignore_edge": ignore_edge,

            "check_negative": check_negative

        }

    }
 
def detect_star_streak_fits(fits_path, hdu_index=0, row_center=None, half_ap=3, **kwargs):

    with fits.open(fits_path, memmap=True) as hdul:

        data = np.asarray(hdul[hdu_index].data)

        if data.ndim != 2:

            raise ValueError("Expected a 2-D reduced image for Case A.")

        x, y = extract_1d_from_2d(data, row_center=row_center, half_ap=half_ap)

    return detect_star_streak_1d(x, y, **kwargs)
 
if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Star-streak detector (Case A: 2-D image -> 1-D extract).")

    ap.add_argument("--fits", required=True, help="Path to reduced 2-D FITS image")

    ap.add_argument("--hdu", type=int, default=0, help="HDU index (default 0)")

    ap.add_argument("--half-ap", type=int, default=3, help="Half aperture height in pixels")

    ap.add_argument("--row-center", type=int, default=None, help="Row index of trace center (optional)")

    ap.add_argument("--sg-window", type=int, default=31)

    ap.add_argument("--sg-poly", type=int, default=3)

    ap.add_argument("--z-prom", type=float, default=6.0)

    ap.add_argument("--w-max", type=int, default=4)

    ap.add_argument("--run-window", type=int, default=200)

    ap.add_argument("--n-min", type=int, default=2)

    ap.add_argument("--s-min", type=float, default=8.0)

    ap.add_argument("--ignore-edge", type=int, default=10)

    ap.add_argument("--no-neg", action="store_true", help="Disable negative spike check")
 
    args = ap.parse_args()

    has_streak, info = detect_star_streak_fits(

        args.fits, hdu_index=args.hdu, row_center=args.row_center, half_ap=args.half_ap,

        sg_window=args.sg_window, sg_poly=args.sg_poly, z_prom=args.z_prom, w_max=args.w_max,

        run_window=args.run_window, n_min=args.n_min, s_min=args.s_min, ignore_edge=args.ignore_edge,

        check_negative=(not args.no_neg)

    )

    print(json.dumps({"has_streak": has_streak, **info}, indent=2))

    # Exit code 0 for clean run; print JSON for CI/automation.

 