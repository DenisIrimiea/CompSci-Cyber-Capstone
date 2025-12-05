# background_gradient.py
# ------------------------------------------------------------
# PURPOSE:
#   Detect if an astronomical image has a large, smooth brightness
#   gradient — like when moonlight or scattered light makes one side
#   of the image brighter than the other.
#
#   Works on the entire 2-D image and checks:
#     • how tilted the background is
#     • if the change is smooth (not random)
#     • if it’s strong enough compared to noise
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation


# -------------------- STEP 1: Measure noise --------------------

def mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = np.median(x)
    return 1.4826 * np.median(np.abs(x - m))


def robust_zscore(img: np.ndarray):
    med = np.median(img)
    s = mad(img) + 1e-9
    return (img - med) / s, med, s


# -------------------- STEP 2: Mask bright stars --------------------

def make_source_mask(img: np.ndarray,
                     z_thresh: float = 3.5,
                     grow: int = 2,
                     extra_dilate: int = 1) -> np.ndarray:

    z, _, _ = robust_zscore(img)
    src = z > float(z_thresh)

    if grow > 0:
        grown = gaussian_filter(src.astype(float), sigma=grow) > 0.05
        src = grown

    if extra_dilate > 0:
        src = binary_dilation(src, iterations=int(extra_dilate))

    return src


# -------------------- STEP 3: Fit background plane --------------------

def fit_plane_irls(img: np.ndarray,
                   mask: np.ndarray | None = None,
                   iters: int = 6,
                   huber_k: float = 1.5):

    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    X = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)], axis=1)
    y = img.ravel()

    use = (~mask).ravel() if mask is not None else np.ones(H * W, dtype=bool)

    w = np.ones_like(y)
    coef = np.array([0.0, 0.0, np.median(y[use])], dtype=float)

    for _ in range(iters):
        Wvec = w[use][:, None]
        XtW = X[use] * Wvec
        coef, *_ = np.linalg.lstsq(XtW, (y[use] * w[use]), rcond=None)

        r = y - X.dot(coef)
        r_bg = r[use]
        s = mad(r_bg) + 1e-9

        t = np.abs(r) / (float(huber_k) * s)
        w = 1.0 / np.maximum(1.0, t)

    resid = (y - X.dot(coef)).reshape(H, W)
    return float(coef[0]), float(coef[1]), float(coef[2]), resid, (~use).reshape(H, W)


# -------------------- STEP 4: Gradient Detector --------------------

def detect_background_gradient(img: np.ndarray,
                               *,
                               z_thresh: float = 3.5,
                               grow: int = 2,
                               extra_dilate: int = 1,
                               min_delta_abs: float = 25.0,        # ↑ updated
                               min_delta_frac: float = 0.15,       # ↑ updated
                               min_delta_vs_noise: float = 2.0,    # ↑ updated
                               vlf_sigma_frac: float = 0.10,
                               min_vlf_var_frac: float = 0.50):    # ↑ updated
    """
    Detects smooth large-scale background gradients.
    """

    img = np.asarray(img, dtype=float)
    H, W = img.shape

    src_mask = make_source_mask(img, z_thresh=z_thresh, grow=grow, extra_dilate=extra_dilate)

    a, b, c, resid, _ = fit_plane_irls(img, mask=src_mask)

    corners = np.array([[0, 0], [0, W-1], [H-1, 0], [H-1, W-1]], dtype=float)
    vals = a * corners[:, 1] + b * corners[:, 0] + c
    plane_delta = float(vals.max() - vals.min())
    slope_mag = float(np.hypot(a, b))

    bg_pix = ~src_mask
    bg_med = float(np.median(img[bg_pix])) if np.any(bg_pix) else float(np.median(img))
    resid_sigma = float(mad(resid[bg_pix])) if np.any(bg_pix) else float(mad(resid))
    delta_frac = plane_delta / (abs(bg_med) + 1e-9)
    strength_vs_noise = plane_delta / (resid_sigma + 1e-9)

    sigma_vlf = max(2.0, vlf_sigma_frac * min(H, W))
    vlf = gaussian_filter(img, sigma=sigma_vlf)
    vlf_var_frac = np.var(vlf) / (np.var(img) + 1e-12)

    passes_abs = plane_delta >= min_delta_abs
    passes_frac = delta_frac >= min_delta_frac
    passes_noise = strength_vs_noise >= min_delta_vs_noise
    passes_vlf = vlf_var_frac >= min_vlf_var_frac

    is_grad = bool(passes_abs and passes_frac and passes_noise and passes_vlf)

    info = {
        "is_background_gradient": is_grad,
        "plane": {"a": a, "b": b, "c": c, "slope_mag": slope_mag},
        "plane_delta_counts": plane_delta,
        "delta_fraction_of_median": delta_frac,
        "residual_sigma": resid_sigma,
        "strength_vs_noise": strength_vs_noise,
        "vlf_sigma": sigma_vlf,
        "vlf_var_fraction": vlf_var_frac,
        "mask_coverage_frac": float(np.mean(src_mask)),
    }
    return is_grad, info


# -------------------- STEP 5: CLI MODE --------------------

if __name__ == "__main__":
    import argparse, json
    from astropy.io import fits

    ap = argparse.ArgumentParser(description="Detect background gradients in a FITS image.")
    ap.add_argument("--fits", required=True)
    ap.add_argument("--hdu", type=int, default=0)
    args = ap.parse_args()

    with fits.open(args.fits, memmap=False) as hdul:
        data = np.asarray(hdul[args.hdu].data, dtype=float)

    ok, info = detect_background_gradient(data)
    print(json.dumps({"background_gradient": ok, **info}, indent=2))
