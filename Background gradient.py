# moon_gradient.py
# ------------------------------------------------------------
# PURPOSE:
#   Detect if an astronomical image has a large, smooth brightness
#   gradient — like when moonlight or scattered light makes one side
#   of the image brighter than the other.
#
# BIG IDEA:
#   - We ignore stars and hot pixels (they’re distractions).
#   - Fit a flat “floor” (a plane) to the background.
#   - If that floor is tilted and the tilt is bigger than noise,
#     we say: “Yep, that’s a moon gradient.”
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
    """
    Median Absolute Deviation (MAD) = a noise estimate that ignores outliers.
    Think: how much do "normal" pixels vary?
    Unlike stddev, it isn’t fooled by bright stars.
    """
    x = np.asarray(x, dtype=float)
    m = np.median(x)  # typical brightness (the “middle” value)
    return 1.4826 * np.median(np.abs(x - m))  # 1.48 scales MAD ≈ stddev for Gaussian noise


def robust_zscore(img: np.ndarray):
    """
    Convert every pixel to a z-score = “how many noise units above average.”
    Example: z = 3 means that pixel is 3× brighter than a normal pixel.
    This helps us find bright stuff (stars, cosmic rays, etc.).
    """
    med = np.median(img)        # overall average brightness (background)
    s = mad(img) + 1e-9         # how much typical pixels vary
    return (img - med) / s, med, s  # z-map, median, noise level


# -------------------- STEP 2: Mask bright stars & junk --------------------

def make_source_mask(img: np.ndarray,
                     z_thresh: float = 3.5,  # threshold: how many σ above average = star
                     grow: int = 2,          # how much to “grow” the mask (in pixels)
                     extra_dilate: int = 1) -> np.ndarray:
    """
    Find and hide stars, cosmic rays, and hot pixels.
    Why? We only want the background when fitting our flat plane.
    """
    z, _, _ = robust_zscore(img)   # get z-scores of all pixels
    src = z > float(z_thresh)      # anything brighter than z=3.5 → likely a star

    # Smooth mask a little so we cover star halos, not just cores
    if grow > 0:
        grown = gaussian_filter(src.astype(float), sigma=grow) > 0.05
        src = grown

    # Expand the mask one more time (makes sure faint edges are covered)
    if extra_dilate > 0:
        src = binary_dilation(src, iterations=int(extra_dilate))

    # Result: True = star/hot pixel (ignore these later)
    return src


# -------------------- STEP 3: Fit a flat plane (background floor) --------------------

def fit_plane_irls(img: np.ndarray,
                   mask: np.ndarray | None = None,
                   iters: int = 6,
                   huber_k: float = 1.5):
    """
    Fit a flat plane: z = a·x + b·y + c
      - 'a' = how much brightness changes left→right
      - 'b' = how much brightness changes top→bottom
      - 'c' = average brightness
    Uses a robust method so stars that sneak through don’t ruin the fit.
    """

    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]              # coordinate grids
    X = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)], axis=1)
    y = img.ravel()

    if mask is None:
        use = np.ones(H * W, dtype=bool)     # use all pixels if no mask
    else:
        use = (~mask).ravel()                # use only background pixels

    w = np.ones_like(y)                      # start with equal weights
    coef = np.array([0.0, 0.0, np.median(y[use])], dtype=float)  # initial guess

    for _ in range(iters):
        # weighted least-squares solve → best-fit plane to background
        Wvec = w[use][:, None]
        XtW = X[use] * Wvec
        coef, *_ = np.linalg.lstsq(XtW, (y[use] * w[use]), rcond=None)

        # residuals = difference between image and fitted plane
        r = y - X.dot(coef)
        r_bg = r[use]
        s = mad(r_bg) + 1e-9

        # Huber weights: give less importance to big outliers (robust)
        t = np.abs(r) / (float(huber_k) * s)
        w = 1.0 / np.maximum(1.0, t)

    resid = (y - X.dot(coef)).reshape(H, W)  # residual image (what’s left after plane)
    return float(coef[0]), float(coef[1]), float(coef[2]), resid, (~use).reshape(H, W)


# -------------------- STEP 4: Detect the gradient --------------------

def detect_moon_gradient(img: np.ndarray,
                         *,
                         z_thresh: float = 3.5,
                         grow: int = 2,
                         extra_dilate: int = 1,
                         min_delta_abs: float = 10.0,
                         min_delta_frac: float = 0.08,
                         min_delta_vs_noise: float = 1.2,
                         vlf_sigma_frac: float = 0.10,
                         min_vlf_var_frac: float = 0.35):
    """
    Main detector function.

    In simple terms:
      1. Mask stars
      2. Fit a flat plane to background
      3. Measure how tilted it is
      4. Blur the image to check if the change is large and smooth
      5. If all tests pass → it's a moon gradient
    """

    img = np.asarray(img, dtype=float)
    H, W = img.shape

    # 1️⃣ Mask stars
    src_mask = make_source_mask(img, z_thresh=z_thresh, grow=grow, extra_dilate=extra_dilate)

    # 2️⃣ Fit flat plane on the background
    a, b, c, resid, _ = fit_plane_irls(img, mask=src_mask, iters=6, huber_k=1.5)

    # 3️⃣ Check how tilted that plane is
    corners = np.array([[0, 0], [0, W - 1], [H - 1, 0], [H - 1, W - 1]], dtype=float)
    vals = a * corners[:, 1] + b * corners[:, 0] + c
    plane_delta = float(vals.max() - vals.min())  # brightness difference corner→corner
    slope_mag = float(np.hypot(a, b))             # how steep it is (counts per pixel)

    # 4️⃣ Figure out if the tilt is big compared to background noise
    bg_pix = ~src_mask
    bg_med = float(np.median(img[bg_pix])) if np.any(bg_pix) else float(np.median(img))
    resid_sigma = float(mad(resid[bg_pix])) if np.any(bg_pix) else float(mad(resid))
    delta_frac = plane_delta / (abs(bg_med) + 1e-9)
    strength_vs_noise = plane_delta / (resid_sigma + 1e-9)

    # 5️⃣ Super-blur the image → keep only slow changes (like moonlight glow)
    sigma_vlf = max(2.0, vlf_sigma_frac * min(H, W))
    vlf = gaussian_filter(img, sigma=sigma_vlf)
    vlf_var_frac = np.var(vlf) / (np.var(img) + 1e-12)

    # 6️⃣ Decision rules (must all pass)
    passes_abs = plane_delta >= min_delta_abs
    passes_frac = delta_frac >= min_delta_frac
    passes_noise = strength_vs_noise >= min_delta_vs_noise
    passes_vlf = vlf_var_frac >= min_vlf_var_frac

    is_grad = bool(passes_abs and passes_frac and passes_noise and passes_vlf)

    # 7️⃣ Return everything for logging or plotting
    info = {
        "is_moon_gradient": is_grad,
        "plane": {
            "a": a, "b": b, "c": c,
            "slope_mag": slope_mag
        },
        "plane_delta_counts": plane_delta,
        "delta_fraction_of_median": delta_frac,
        "residual_sigma": resid_sigma,
        "strength_vs_noise": strength_vs_noise,
        "vlf_sigma": sigma_vlf,
        "vlf_var_fraction": vlf_var_frac,
        "mask_coverage_frac": float(np.mean(src_mask)),
    }
    return is_grad, info


# -------------------- STEP 5: Command-line mode --------------------

if __name__ == "__main__":
    """
    This lets you run the script directly:
      python moon_gradient.py --fits my_image.fits

    It loads the FITS image, runs the detector, and prints a summary.
    """
    import argparse, json
    from astropy.io import fits

    ap = argparse.ArgumentParser(description="Detect moonlight gradient in a FITS image.")
    ap.add_argument("--fits", required=True, help="Path to your FITS file")
    ap.add_argument("--hdu", type=int, default=0, help="Which HDU to read (default=0)")
    args = ap.parse_args()

    with fits.open(args.fits, memmap=False) as hdul:
        data = np.asarray(hdul[args.hdu].data, dtype=float)

    ok, info = detect_moon_gradient(data)
    print(json.dumps({"moon_gradient": ok, **info}, indent=2))
