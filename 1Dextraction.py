#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def select_first_box(flux, elong, i0, j0, box_gap, col_tol):
    nbh, nbw = flux.shape
    cand = np.zeros_like(flux, dtype=bool)

    for i in range(nbh):
        for j in range(nbw):
            if abs(i - i0) >= box_gap and abs(j - j0) <= col_tol:
                cand[i, j] = True

    if not cand.any():
        cand[:, :] = True
        for i in range(nbh):
            for j in range(nbw):
                if abs(i - i0) < box_gap:
                    cand[i, j] = False

    elong_boost = np.maximum(elong - 1.0, 0.0)
    score = flux * elong_boost
    masked = np.where(cand, score, -np.inf)
    flat_idx = np.argmax(masked)
    i1, j1 = np.unravel_index(flat_idx, flux.shape)
    return int(i1), int(j1)


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
        f"projection({x1_ds9:.6f},{y1_ds9:.6f},{x2_ds9:.6f},{y2_ds9:.6f},{width})\n"
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


def main():
    p = argparse.ArgumentParser(
        description="Shape-aware zeroth/first detection + DS9-like angled extraction + first-order overexposure flag."
    )
    p.add_argument("--fits", required=True)
    p.add_argument("--box-pix", type=int, default=32)
    p.add_argument("--box-gap", type=int, default=2)
    p.add_argument("--col-tol", type=int, default=1)
    p.add_argument("--width", type=int, default=1)
    p.add_argument("--pad-frac", type=float, default=0.1)
    p.add_argument("--sat-level", type=float, default=65535.0)
    p.add_argument("--sat-margin", type=float, default=1000.0)
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
    print(f"zeroth tile (i0,j0)=({i0_box},{j0_box}), center≈(x0,y0)=({x0:.1f},{y0:.1f})")

    i1_box, j1_box = select_first_box(flux, elong, i0_box, j0_box,
                                      args.box_gap, args.col_tol)
    y1, x1 = box_center(i1_box, j1_box, args.box_pix, args.box_pix)
    print(f"first  tile (i1,j1)=({i1_box},{j1_box}), center≈(x1,y1)=({x1:.1f},{y1:.1f})")

    overexp, info = detect_overexposure_first(
        data, i0_box, j0_box, i1_box, j1_box,
        args.box_pix, args.box_pix,
        sat_level=args.sat_level, margin=args.sat_margin
    )
    print(f"FIRST-ORDER OVEREXPOSED: {overexp} "
          f"(max0={info['max_zeroth']:.1f}, max1={info['max_first']:.1f}, "
          f"thr={info['thr']:.1f}, "
          f"near_sat={info['first_near_sat']}, "
          f"first>zeroth={info['first_brighter_than_zeroth']})")

    dx = x1 - x0
    dy = y1 - y0
    length = float(np.hypot(dx, dy))
    ux = dx / length
    uy = dy / length

    pad = args.pad_frac * length
    x_start = x0 - ux * pad
    y_start = y0 - uy * pad
    x_start = float(np.clip(x_start, 0, w - 1))
    y_start = float(np.clip(y_start, 0, h - 1))

    x_end, y_end = line_to_frame_end(x_start, y_start, ux, uy, h, w)

    print(f"line from ({x_start:.1f},{y_start:.1f}) to ({x_end:.1f},{y_end:.1f}), "
          f"width={args.width}")

    s, profile = extract_along_line(
        data, x_start, y_start, x_end, y_end,
        width=args.width, n_samples=None, mode="sum"
    )

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
    plt.title("Shape-aware angled zeroth→first extraction")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"wrote spectrum plot to {out_png}")

    write_ds9_projection(out_reg, x_start, y_start, x_end, y_end, args.width)
    print(f"wrote DS9 projection region to {out_reg}")


if __name__ == "__main__":
    main()
