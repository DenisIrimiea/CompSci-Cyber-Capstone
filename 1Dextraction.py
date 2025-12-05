#!/usr/bin/env python3
 
import argparse
from pathlib import Path
 
import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
 
def mad(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))
 
 
# -------- tile-level stats --------
 
def tile_stats(img, box_h, box_w):
    h, w = img.shape
    nbh = h // box_h
    nbw = w // box_w
    img_crop = img[:nbh * box_h, :nbw * box_w]
 
    flux = np.zeros((nbh, nbw), dtype=float)
    elong = np.ones((nbh, nbw), dtype=float)
 
    y_idx = np.arange(box_h)[:, None]
    x_idx = np.arange(box_w)[None, :]
 
    for i in range(nbh):
        for j in range(nbw):
            tile = img_crop[i * box_h:(i + 1) * box_h,
                            j * box_w:(j + 1) * box_w]
            S = tile.sum()
            flux[i, j] = S
            if S <= 0:
                continue
 
            yc = (tile * y_idx).sum() / S
            xc = (tile * x_idx).sum() / S
 
            dy = y_idx - yc
            dx = x_idx - xc
 
            Cyy = (tile * dy * dy).sum() / S
            Cxx = (tile * dx * dx).sum() / S
            Cxy = (tile * dy * dx).sum() / S
 
            cov = np.array([[Cxx, Cxy],
                            [Cxy, Cyy]])
            vals = np.linalg.eigvalsh(cov)
            lam1 = max(vals[0], 1e-6)
            lam2 = max(vals[1], lam1)
            elong[i, j] = lam2 / lam1
 
    return flux, elong, nbh, nbw
 
 
def box_center(i_box, j_box, box_h, box_w):
    r0 = i_box * box_h + box_h / 2.0
    c0 = j_box * box_w + box_w / 2.0
    return float(r0), float(c0)
 
 
def box_slice(data, i_box, j_box, box_h, box_w):
    h, w = data.shape
    r1 = i_box * box_h
    r2 = min((i_box + 1) * box_h, h)
    c1 = j_box * box_w
    c2 = min((j_box + 1) * box_w, w)
    return data[r1:r2, c1:c2]
 
 
def select_zeroth_box(flux, elong):
    R = np.maximum(elong, 1.0)
    score = flux / R
    flat_idx = np.argmax(score)
    i0, j0 = np.unravel_index(flat_idx, flux.shape)
    return int(i0), int(j0)
 
 
def select_branch_candidates(flux, elong, i0, j0, box_gap, col_tol):
    R = np.maximum(elong, 1.0)
    score = flux / R
 
    nbh, nbw = flux.shape
    upper = np.zeros_like(flux, dtype=bool)
    lower = np.zeros_like(flux, dtype=bool)
 
    for i in range(nbh):
        for j in range(nbw):
            if abs(j - j0) <= col_tol:
                if i <= i0 - box_gap:
                    upper[i, j] = True
                elif i >= i0 + box_gap:
                    lower[i, j] = True
 
    def best(mask):
        if not mask.any():
            return None
        masked = np.where(mask, score, -np.inf)
        idx = np.argmax(masked)
        if not np.isfinite(masked.flat[idx]):
            return None
        i, j = np.unravel_index(idx, score.shape)
        return int(i), int(j)
 
    cand_up = best(upper)
    cand_lo = best(lower)
    return cand_up, cand_lo
 
 
# -------- extraction geometry --------
 
def extract_along_line(img, x1, y1, x2, y2, width=1, n_samples=None, mode="sum"):
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))
    if length == 0:
        raise ValueError("zero-length line")
    if n_samples is None:
        n_samples = int(np.ceil(length))
 
    ux = dx / length
    uy = dy / length
 
    px = -uy
    py = ux
 
    t = np.linspace(0.0, length, n_samples)
    x_center = x1 + ux * t
    y_center = y1 + uy * t
 
    if width <= 1:
        coords = np.vstack([y_center, x_center])
        vals = map_coordinates(img, coords, order=1, mode="nearest")
        return t, vals
 
    offsets = np.arange(-(width - 1) / 2.0, (width + 1) / 2.0)
    samples = []
    for o in offsets:
        xs = x_center + px * o
        ys = y_center + py * o
        coords = np.vstack([ys, xs])
        vals = map_coordinates(img, coords, order=1, mode="nearest")
        samples.append(vals)
    samples = np.stack(samples, axis=0)
 
    if mode == "sum":
        prof = samples.sum(axis=0)
    elif mode == "mean":
        prof = samples.mean(axis=0)
    else:
        raise ValueError("mode must be 'sum' or 'mean'")
    return t, prof
 
 
def write_ds9_projection(path, x1, y1, x2, y2, width):
    x1_ds9 = x1 + 1.0
    y1_ds9 = y1 + 1.0
    x2_ds9 = x2 + 1.0
    y2_ds9 = y2 + 1.0
    txt = (
        "# Region file format: DS9 version 4.1\n"
        "physical\n"
        f"projection({x1_ds9:.6f},{y1_ds9:.6f},"
        f"{x2_ds9:.6f},{y2_ds9:.6f},{width})\n"
    )
    path.write_text(txt)
 
 
def line_to_frame_end(x_start, y_start, ux, uy, h, w):
    s_candidates = []
    if ux > 0:
        s_candidates.append((w - 1 - x_start) / ux)
    elif ux < 0:
        s_candidates.append((0 - x_start) / ux)
    if uy > 0:
        s_candidates.append((h - 1 - y_start) / uy)
    elif uy < 0:
        s_candidates.append((0 - y_start) / uy)
    s_candidates = [s for s in s_candidates if s > 0]
    if not s_candidates:
        return x_start, y_start
    s_max = min(s_candidates)
    x_end = x_start + ux * s_max
    y_end = y_start + uy * s_max
    return x_end, y_end
 
 
def refine_line(img, x0, y0, x1, y1, width=1,
                search_radius=15, probe_samples=400,
                probe_center_frac=0.8, probe_half_frac=0.15):
    dx = x1 - x0
    dy = y1 - y0
    length = float(np.hypot(dx, dy))
    ux = dx / length
    uy = dy / length
 
    px = -uy
    py = ux
 
    best_score = -np.inf
    best_y0, best_y1, best_x0, best_x1 = y0, y1, x0, x1
 
    for o in range(-search_radius, search_radius + 1):
        xs0 = x0 + px * o
        ys0 = y0 + py * o
        xs1 = x1 + px * o
        ys1 = y1 + py * o
 
        t, prof = extract_along_line(
            img, xs0, ys0, xs1, ys1,
            width=width, n_samples=probe_samples, mode="sum"
        )
 
        n = len(prof)
        if n < 10:
            continue
 
        c_idx = int(probe_center_frac * n)
        hw = int(probe_half_frac * n)
        i0 = max(0, c_idx - hw)
        i1 = min(n, c_idx + hw)
 
        seg = prof[i0:i1]
        if seg.size == 0:
            continue
        med = np.median(seg)
        peak = np.max(seg)
        score = peak - med
 
        if score > best_score:
            best_score = score
            best_x0, best_y0 = xs0, ys0
            best_x1, best_y1 = xs1, ys1
 
    return best_x0, best_y0, best_x1, best_y1
 
 
# -------- branch scoring (peak-based) --------
 
def branch_score(data, x0, y0, i1_box, j1_box,
                 box_pix, width, pad_frac, h, w,
                 margin_pix=30, sg_window=101, high_sigma=2.0):
    if (i1_box is None) or (j1_box is None):
        return -np.inf
 
    y1_c, x1_c = box_center(i1_box, j1_box, box_pix, box_pix)
 
    dx = x1_c - x0
    dy = y1_c - y0
    L = float(np.hypot(dx, dy))
    if L < 5.0:
        return -np.inf
 
    pad = pad_frac * L
 
    x0_ref, y0_ref, x1_ref, y1_ref = refine_line(
        data, x0, y0, x1_c, y1_c, width=width
    )
 
    dx_ref = x1_ref - x0_ref
    dy_ref = y1_ref - y0_ref
    len_ref = float(np.hypot(dx_ref, dy_ref))
    if len_ref <= 0:
        return -np.inf
 
    ux_ref = dx_ref / len_ref
    uy_ref = dy_ref / len_ref
 
    x_start = x0_ref - ux_ref * pad
    y_start = y0_ref - uy_ref * pad
    x_start = float(np.clip(x_start, 0, w - 1))
    y_start = float(np.clip(y_start, 0, h - 1))
 
    x_end, y_end = line_to_frame_end(x_start, y_start, ux_ref, uy_ref, h, w)
 
    t, prof = extract_along_line(
        data, x_start, y_start, x_end, y_end,
        width=width, n_samples=None, mode="sum"
    )
 
    n = len(prof)
    if n < 100:
        return -np.inf
 
    vx = x0_ref - x_start
    vy = y0_ref - y_start
    t_zero = vx * ux_ref + vy * uy_ref
    i_zero = int(np.argmin(np.abs(t - t_zero)))
 
    t_cut = t_zero + margin_pix
    first_start = int(np.searchsorted(t, t_cut))
    if first_start >= n - 20:
        return -np.inf
 
    first_t = t[first_start:]
    first_p = prof[first_start:]
    m = first_p.size
    if m < 20:
        return -np.inf
 
    wlen = min(sg_window, m - 1 if (m % 2 == 0) else m)
    if wlen < 7:
        smooth = first_p
    else:
        if wlen % 2 == 0:
            wlen += 1
        smooth = savgol_filter(first_p, window_length=wlen,
                               polyorder=3, mode="interp")
 
    tail = smooth[int(0.7 * m):]
    if tail.size < 5:
        tail = smooth
    bg = float(np.median(tail))
 
    signal = smooth - bg
    sigma_loc = mad(signal) or (np.std(signal) + 1e-9)
 
    high = signal > high_sigma * sigma_loc
    if not np.any(high):
        return -np.inf
 
    idx = np.where(high)[0]
    if idx.size == 0:
        return -np.inf
 
    splits = np.where(np.diff(idx) > 1)[0] + 1
    segments = np.split(idx, splits)
 
    best_score = -np.inf
    for seg in segments:
        if seg.size == 0:
            continue
        seg_width = seg.size
        seg_intensity = float(signal[seg].mean())
        score = seg_width * seg_intensity
        if score > best_score:
            best_score = score
 
    return best_score
 
 
# -------- flags: overexposure + low SNR --------
 
def detect_overexposure_first(data, i0_box, j0_box, i1_box, j1_box,
                              box_h, box_w,
                              sat_level=65535.0, margin=1000.0):
    z_patch = box_slice(data, i0_box, j0_box, box_h, box_w)
    f_patch = box_slice(data, i1_box, j1_box, box_h, box_w)
 
    max0 = float(z_patch.max())
    max1 = float(f_patch.max())
 
    thr = sat_level - margin
    first_near_sat = max1 >= thr
    first_brighter = max1 > max0
 
    is_over = first_near_sat or first_brighter
    info = {
        "max_zeroth": max0,
        "max_first": max1,
        "thr": thr,
        "first_near_sat": first_near_sat,
        "first_brighter_than_zeroth": first_brighter,
    }
    return is_over, info
 
 
def compute_snr_metrics(profile, sg_window=51, sg_poly=3):
    n = len(profile)
    if n < 7:
        return {
            "sigma": float("nan"),
            "dyn_snr": float("nan"),
            "peak_snr": float("nan"),
            "coverage": 0.0,
        }
 
    if sg_window >= n:
        sg_window = (n - 1) | 1
    if sg_window % 2 == 0:
        sg_window += 1
 
    smooth = savgol_filter(profile, window_length=sg_window,
                           polyorder=sg_poly, mode="interp")
    resid = profile - smooth
    sigma = mad(resid) or (np.std(resid) + 1e-9)
 
    p95 = np.percentile(smooth, 95.0)
    p5 = np.percentile(smooth, 5.0)
    dyn_snr = (p95 - p5) / (6.0 * sigma)
 
    med = float(np.median(smooth))
    peak_snr = (smooth.max() - med) / sigma
 
    high = (smooth - med) >= 3.0 * sigma
    coverage = float(high.mean())
 
    return {
        "sigma": float(sigma),
        "dyn_snr": float(dyn_snr),
        "peak_snr": float(peak_snr),
        "coverage": coverage,
    }
 
 
# -------- main CLI --------
 
def main():
    p = argparse.ArgumentParser(
        description=(
            "Shape-aware zeroth/first detection, peak-based branch selection, "
            "first-order overexposure flag, and low-SNR flag."
        )
    )
    p.add_argument("--fits", required=True)
    p.add_argument("--box-pix", type=int, default=32)
    p.add_argument("--box-gap", type=int, default=2)
    p.add_argument("--col-tol", type=int, default=1)
    p.add_argument("--width", type=int, default=1)
    p.add_argument("--pad-frac", type=float, default=0.1)
    p.add_argument("--sat-level", type=float, default=65535.0)
    p.add_argument("--sat-margin", type=float, default=1000.0)
    p.add_argument("--dyn-thresh", type=float, default=8.0)
    p.add_argument("--peak-thresh", type=float, default=10.0)
    p.add_argument("--cov-thresh", type=float, default=0.01)
    p.add_argument("--prefix")
    args = p.parse_args()
 
    fits_path = Path(args.fits)
    prefix = args.prefix or fits_path.stem
 
    out_txt = Path(f"{prefix}_line_1d.txt")
    out_png = Path(f"{prefix}_line_spectrum.png")
    out_reg = Path(f"{prefix}_line_projection.reg")
 
    with fits.open(fits_path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        if data.ndim != 2:
            raise ValueError("Expected 2-D image in HDU 0.")
    h, w = data.shape
 
    flux, elong, nbh, nbw = tile_stats(data, args.box_pix, args.box_pix)
 
    i0_box, j0_box = select_zeroth_box(flux, elong)
    y0, x0 = box_center(i0_box, j0_box, args.box_pix, args.box_pix)
    print(f"zeroth tile (i0,j0)=({i0_box},{j0_box}), center~(x0,y0)=({x0:.1f},{y0:.1f})")
 
    cand_up, cand_lo = select_branch_candidates(
        flux, elong, i0_box, j0_box, args.box_gap, args.col_tol
    )
 
    score_up = branch_score(
        data, x0, y0,
        *(cand_up if cand_up is not None else (None, None)),
        box_pix=args.box_pix,
        width=args.width,
        pad_frac=args.pad_frac,
        h=h, w=w,
    )
    score_lo = branch_score(
        data, x0, y0,
        *(cand_lo if cand_lo is not None else (None, None)),
        box_pix=args.box_pix,
        width=args.width,
        pad_frac=args.pad_frac,
        h=h, w=w,
    )
 
    print(f"branch scores (peak-based SNR): upper={score_up:.2f}, lower={score_lo:.2f}")
 
    if (score_up <= -np.inf) and (score_lo <= -np.inf):
        if cand_up is not None:
            i1_box, j1_box = cand_up
        elif cand_lo is not None:
            i1_box, j1_box = cand_lo
        else:
            i1_box, j1_box = i0_box, j0_box
    elif score_up >= score_lo:
        i1_box, j1_box = cand_up
    else:
        i1_box, j1_box = cand_lo
 
    y1, x1 = box_center(i1_box, j1_box, args.box_pix, args.box_pix)
    side = "upper" if i1_box < i0_box else "lower"
    print(f"first  tile (i1,j1)=({i1_box},{j1_box}), side={side}, center~(x1,y1)=({x1:.1f},{y1:.1f})")
 
    overexp, info_over = detect_overexposure_first(
        data, i0_box, j0_box, i1_box, j1_box,
        args.box_pix, args.box_pix,
        sat_level=args.sat_level, margin=args.sat_margin
    )
    print(
        "FIRST-ORDER OVEREXPOSED: {over} "
        "(max0={m0:.1f}, max1={m1:.1f}, thr={thr:.1f}, "
        "near_sat={ns}, first>zeroth={fz})".format(
            over=overexp,
            m0=info_over["max_zeroth"],
            m1=info_over["max_first"],
            thr=info_over["thr"],
            ns=info_over["first_near_sat"],
            fz=info_over["first_brighter_than_zeroth"],
        )
    )
 
    dx = x1 - x0
    dy = y1 - y0
    length = float(np.hypot(dx, dy))
    ux = dx / length
    uy = dy / length
    pad = args.pad_frac * length
 
    x0_ref, y0_ref, x1_ref, y1_ref = refine_line(
        data, x0, y0, x1, y1, width=args.width
    )
 
    dx_ref = x1_ref - x0_ref
    dy_ref = y1_ref - y0_ref
    len_ref = float(np.hypot(dx_ref, dy_ref))
    ux_ref = dx_ref / len_ref
    uy_ref = dy_ref / len_ref
 
    x_start = x0_ref - ux_ref * pad
    y_start = y0_ref - uy_ref * pad
    x_start = float(np.clip(x_start, 0, w - 1))
    y_start = float(np.clip(y_start, 0, h - 1))
 
    x_end, y_end = line_to_frame_end(x_start, y_start, ux_ref, uy_ref, h, w)
 
    print(
        f"line (refined) from ({x_start:.1f},{y_start:.1f}) "
        f"to ({x_end:.1f},{y_end:.1f}), width={args.width}"
    )
 
    s, profile = extract_along_line(
        data, x_start, y_start, x_end, y_end,
        width=args.width, n_samples=None, mode="sum"
    )
 
    snr = compute_snr_metrics(profile)
    print(
        "SNR metrics: sigma={sig:.2f}, dyn_snr={dyn:.2f}, "
        "peak_snr={pk:.2f}, coverage={cov:.3f}".format(
            sig=snr["sigma"],
            dyn=snr["dyn_snr"],
            pk=snr["peak_snr"],
            cov=snr["coverage"],
        )
    )
 
    low_snr = (
        (snr["dyn_snr"] < args.dyn_thresh) or
        (snr["peak_snr"] < args.peak_thresh) or
        (snr["coverage"] < args.cov_thresh)
    )
    print(f"LOW_SNR: {low_snr}")
 
    arr = np.column_stack((s, profile))
    np.savetxt(
        out_txt,
        arr,
        fmt="%.6f",
        header="s counts   # s = distance along line (pixels)",
        comments="",
    )
    print(f"wrote 1-D line spectrum to {out_txt}")
 
    plt.figure(figsize=(8, 5))
    plt.plot(s, profile, lw=1)
    plt.xlabel("Distance along line (pixels)")
    plt.ylabel("Counts")
    plt.title("Shape-aware angled zeroth->first extraction")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"wrote spectrum plot to {out_png}")
 
    write_ds9_projection(out_reg, x_start, y_start, x_end, y_end, args.width)
    print(f"wrote DS9 projection region to {out_reg}")
 
 
if __name__ == "__main__":
    main()
