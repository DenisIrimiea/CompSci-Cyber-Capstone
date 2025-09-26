#!/usr/bin/env python3
"""
Star Streak Detection – SG-profiles, Peaks-first, and Derivative voting
(Updated with an explicit classifier verdict: streak_present True/False)

Quick start:
  pip install numpy scipy astropy matplotlib
  python star_streak_pipeline.py --fits your_spectrum.fits --out out_prefix

Outputs:
  - JSON summary (stdout and <out>_summary.json)
  - PNG plot of detections (<out>_plot.png)
  - CSV of peak candidates (<out>_peaks.csv)

Works with:
  - 2-D reduced images (rows/cols are scanned for peaks/edges)
  - 1-D spectra (promoted internally to a single-row image)
"""
from __future__ import annotations
import argparse
import json
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

try:
    from astropy.io import fits
except Exception:  # optional until you need FITS
    fits = None

# =========================
# Utility & robust stats
# =========================

def _mad(a: np.ndarray) -> float:
    m = np.median(a)
    return float(np.median(np.abs(a - m)) + 1e-12)

def _robust_sigma(y: np.ndarray, win: Optional[int] = None, poly: int = 3) -> Tuple[float, np.ndarray, np.ndarray]:
    n = len(y)
    if win is None:
        win = max(31, (n // 100) * 2 + 1)
    win = min(max(5, win | 1), n - 1 - ((n - 1) % 2))
    trend = savgol_filter(y, window_length=win, polyorder=poly, mode="interp")
    resid = y - trend
    sigma = 1.4826 * _mad(resid)
    return float(sigma), trend, resid

# =========================
# Core data containers
# =========================

class ReducedImage:
    """Wraps a reduced (noise-reduced, spectrally processed) 2-D image.
    Can also wrap a 1-D spectrum by expanding dims.
    """
    def __init__(self, img: np.ndarray, meta: Optional[Dict[str, Any]] = None):
        if img.ndim == 1:
            img = img[None, :]
        if img.ndim != 2:
            raise ValueError("ReducedImage expects 2D (or 1D which is promoted to 2D)")
        self.img = img.astype(float)
        self.meta = meta or {}

    def toArray(self) -> np.ndarray:
        return self.img

    def roi(self, box: Tuple[int, int, int, int]) -> "ReducedImage":
        x0, y0, x1, y1 = box
        return ReducedImage(self.img[y0:y1, x0:x1], self.meta)

    def shape(self) -> Tuple[int, int]:
        return self.img.shape

class ImageMetadata:
    def __init__(self, meta: Dict[str, Any]):
        self.meta = meta

    def get(self, key: str, default: Any = None) -> Any:
        return self.meta.get(key, default)

    def keys(self):
        return self.meta.keys()

# =========================
# Profile sampling
# =========================

class Profile:
    def __init__(self, x: np.ndarray, y: np.ndarray, axis: str, index: int):
        self.x = x.astype(float)
        self.y = y.astype(float)
        self.axis = axis  # 'row' or 'col' or 'path'
        self.index = index

class ProfileSampler:
    def __init__(self, reduced: ReducedImage, x_scale_nm_per_pix: Optional[float] = None):
        self.reduced = reduced
        self.x_scale = x_scale_nm_per_pix  # if you want wavelength per pixel for columns

    def row(self, y: int) -> Profile:
        h, w = self.reduced.shape()
        y = int(np.clip(y, 0, h - 1))
        arr = self.reduced.toArray()[y, :]
        x = np.arange(w)
        if self.x_scale:
            x = x * self.x_scale
        return Profile(x, arr, axis="row", index=y)

    def col(self, x: int) -> Profile:
        h, w = self.reduced.shape()
        x = int(np.clip(x, 0, w - 1))
        arr = self.reduced.toArray()[:, x]
        axis_x = np.arange(h)
        if self.x_scale:
            axis_x = axis_x * self.x_scale
        return Profile(axis_x, arr, axis="col", index=x)

    def grid(self, step: int = 16) -> List[Profile]:
        h, w = self.reduced.shape()
        out: List[Profile] = []
        for y in range(0, h, max(1, step)):
            out.append(self.row(y))
        for x in range(0, w, max(1, step)):
            out.append(self.col(x))
        return out

# =========================
# Savitzky–Golay filter
# =========================

class SavitzkyGolayFilter:
    def smooth(self, profile: Profile, window: int, poly: int) -> Profile:
        n = len(profile.y)
        window = min(max(poly + 3, window | 1), max(5, n - 1 - ((n - 1) % 2)))
        y_s = savgol_filter(profile.y, window_length=window, polyorder=poly, mode="interp")
        return Profile(profile.x, y_s, profile.axis, profile.index)

    def derivative(self, profile: Profile, order: int, window: int, poly: int) -> Profile:
        n = len(profile.y)
        window = min(max(poly + 3, window | 1), max(5, n - 1 - ((n - 1) % 2)))
        # We compute derivative wrt index and convert to per-x using dx
        dy_di = savgol_filter(profile.y, window_length=window, polyorder=poly, deriv=order, delta=1.0, mode="interp")
        dx_di = np.gradient(profile.x)
        denom = (dx_di ** order)
        denom[np.abs(denom) < 1e-12] = 1e-12
        d_ord = dy_di / denom
        return Profile(profile.x, d_ord, profile.axis, profile.index)

# =========================
# Peak detection & derivative edges
# =========================

@dataclass
class Peak:
    index: int
    x: float
    y: float
    snr: float
    fwhm: float

class PeakDetector:
    def find(self, profile: Profile, minProminence: float, minDistance: float, sigma_mode: str = "MAD") -> List[Peak]:
        n = len(profile.y)
        window = min(max(7, (n // 50) * 2 + 1), n - 1 - ((n - 1) % 2))
        y_s = savgol_filter(profile.y, window_length=window, polyorder=3, mode="interp")
        sigma, trend, _ = _robust_sigma(y_s, win=window, poly=3)
        height_thr = np.median(trend) + minProminence * sigma
        dx_med = float(np.median(np.gradient(profile.x)))
        dist_samples = max(1, int(round(minDistance / max(dx_med, 1e-12))))
        pk_idx, props = find_peaks(y_s, height=height_thr, prominence=minProminence * sigma, distance=dist_samples)
        peaks: List[Peak] = []
        if pk_idx.size:
            w_samples, *_ = peak_widths(y_s, pk_idx, rel_height=0.5)
            fwhm = w_samples * dx_med
            snr = (props["peak_heights"] - np.median(trend)) / (sigma + 1e-12)
            for i, k in enumerate(pk_idx):
                peaks.append(Peak(index=int(k), x=float(profile.x[k]), y=float(y_s[k]), snr=float(snr[i]), fwhm=float(fwhm[i])))
        return peaks

class DerivativeZeroCrossing:
    def edges(self, derivProfile: Profile, threshold: float) -> List[int]:
        z = derivProfile.y
        sig = 1.4826 * _mad(z)
        if sig <= 0:
            sig = np.std(z) + 1e-12
        mask = np.abs(z) >= (threshold * sig)
        idx = np.flatnonzero(mask)
        keep: List[int] = []
        for i in idx:
            if i == 0 or i >= len(z) - 1:
                continue
            if z[i - 1] * z[i + 1] < 0 or np.sign(z[i - 1]) != np.sign(z[i]):
                keep.append(int(i))
        return keep

# =========================
# Streak detection (aggregate across profiles)
# =========================

@dataclass
class LineSegment:
    p0: Tuple[int, int]
    p1: Tuple[int, int]
    strength: float

class StreakDetector:
    """Detect line-like streaks by aggregating peak/edge candidates across many row/col profiles."""
    def __init__(self, sampler: ProfileSampler):
        self.sampler = sampler
        self.sg = SavitzkyGolayFilter()
        self.peaks = PeakDetector()
        self.dzc = DerivativeZeroCrossing()

    def detect(self, img: ReducedImage,
               row_step: int = 8,
               col_step: int = 32,
               peak_prom_sigma: float = 6.0,
               peak_min_dist: float = 2.0,
               deriv_sigma_k: float = 6.0) -> List[LineSegment]:
        H, W = img.shape()
        segments: List[LineSegment] = []

        # Horizontal scan (rows) — peak clusters
        for y in range(0, H, max(1, row_step)):
            prof = self.sampler.row(y)
            pk = self.peaks.find(prof, minProminence=peak_prom_sigma, minDistance=peak_min_dist)
            if pk:
                xs = sorted([p.index for p in pk])
                run_start = xs[0]
                last = xs[0]
                for xi in xs[1:] + [xs[-1] + 1000]:
                    if xi <= last + 2:  # allow small gaps
                        last = xi
                        continue
                    x0 = run_start
                    x1 = last
                    strength = float(np.mean([p.snr for p in pk if x0 <= p.index <= x1]))
                    segments.append(LineSegment(p0=(x0, y), p1=(x1, y), strength=strength))
                    run_start = xi
                    last = xi

        # Vertical scan (cols) — derivative edges
        for x in range(0, W, max(1, col_step)):
            prof = self.sampler.col(x)
            d1 = self.sg.derivative(prof, order=1, window=31, poly=3)
            edges = self.dzc.edges(d1, threshold=deriv_sigma_k)
            if edges:
                ys = sorted(edges)
                run_start = ys[0]
                last = ys[0]
                for yi in ys[1:] + [ys[-1] + 1000]:
                    if yi <= last + 2:
                        last = yi
                        continue
                    strength = float(np.mean(np.abs(d1.y[run_start:last + 1])))
                    segments.append(LineSegment(p0=(x, run_start), p1=(x, last), strength=strength))
                    run_start = yi
                    last = yi

        return segments

    def score(self, img: ReducedImage) -> float:
        segs = self.detect(img)
        if not segs:
            return 0.0
        strengths = np.array([s.strength for s in segs], float)
        return float(np.clip(len(segs) / 10.0 + strengths.mean() / 10.0, 0.0, 1.0))

# =========================
# Background / Features / Classifier
# =========================

class BackgroundEstimator:
    def trend(self, profile: Profile, window: int = 151, poly: int = 2) -> Profile:
        n = len(profile.y)
        window = min(max(poly + 3, window | 1), max(5, n - 1 - ((n - 1) % 2)))
        y_t = savgol_filter(profile.y, window_length=window, polyorder=poly, mode="interp")
        return Profile(profile.x, y_t, profile.axis, profile.index)

    def slope(self, img: ReducedImage) -> float:
        arr = img.toArray().astype(float)
        dx = np.mean(np.abs(np.diff(arr, axis=1))) if arr.shape[1] > 1 else 0.0
        dy = np.mean(np.abs(np.diff(arr, axis=0))) if arr.shape[0] > 1 else 0.0
        return float(dx + dy)

    def remove(self, img: ReducedImage, window: int = 151, poly: int = 2) -> ReducedImage:
        arr = img.toArray().astype(float)
        out = arr.copy()
        for y in range(arr.shape[0]):
            prof = Profile(np.arange(arr.shape[1]), arr[y], axis="row", index=y)
            t = self.trend(prof, window=window, poly=poly).y
            out[y] = arr[y] - t
        return ReducedImage(out, img.meta)

@dataclass
class FeatureVector:
    n_segments: int
    mean_strength: float
    max_strength: float
    bg_slope: float
    total_peaks: int

    def get(self, name: str) -> Any:
        return getattr(self, name)

    def toDict(self) -> Dict[str, Any]:
        return {
            "n_segments": self.n_segments,
            "mean_strength": self.mean_strength,
            "max_strength": self.max_strength,
            "bg_slope": self.bg_slope,
            "total_peaks": self.total_peaks,
        }

class FeatureExtractor:
    def __init__(self, sampler: ProfileSampler):
        self.sampler = sampler
        self.sg = SavitzkyGolayFilter()
        self.peaks = PeakDetector()
        self.bg = BackgroundEstimator()

    def extract(self, img: ReducedImage, streaks: List[LineSegment], bg: BackgroundEstimator) -> FeatureVector:
        nseg = len(streaks)
        strengths = np.array([s.strength for s in streaks], float) if streaks else np.array([0.0])
        bg_slope = bg.slope(img)
        total_peaks = 0
        for prof in self.sampler.grid(step=max(1, img.shape()[0] // 16)):
            pk = self.peaks.find(prof, minProminence=6.0, minDistance=2.0)
            total_peaks += len(pk)
        return FeatureVector(
            n_segments=nseg,
            mean_strength=float(strengths.mean()),
            max_strength=float(strengths.max()),
            bg_slope=float(bg_slope),
            total_peaks=int(total_peaks),
        )

class RuleBasedClassifier:
    class Label:
        GREEN = "GREEN"   # No streak indicators
        YELLOW = "YELLOW" # Possible/weak streak indicators
        RED   = "RED"     # Strong streak indicators (treat as streak present)

    def classify(self, fv: FeatureVector) -> str:
        # Tune these thresholds for your dataset
        if fv.n_segments >= 6 or fv.total_peaks >= 50 or fv.max_strength > 12:
            return self.Label.RED
        if fv.n_segments >= 2 or fv.total_peaks >= 15:
            return self.Label.YELLOW
        return self.Label.GREEN

    def streak_present_bool(self, label: str) -> bool:
        # Treat YELLOW and RED as "True" if you want a sensitive detector; or RED-only if conservative.
        return label in {self.Label.YELLOW, self.Label.RED}

    def explain(self, fv: FeatureVector) -> str:
        return (
            f"segments={fv.n_segments}, total_peaks={fv.total_peaks}, "
            f"max_strength={fv.max_strength:.2f}, mean_strength={fv.mean_strength:.2f}, bg_slope={fv.bg_slope:.2f}"
        )

@dataclass
class Result:
    label: str
    streak_present: bool
    features: FeatureVector
    streaks: List[LineSegment]
    meta: Dict[str, Any]

    def summary(self) -> str:
        return (f"Streak present: {self.streak_present} | "
                f"Label={self.label} | {RuleBasedClassifier().explain(self.features)}")

    def toJSON(self) -> str:
        payload = {
            "streak_present": self.streak_present,
            "label": self.label,
            "features": self.features.toDict(),
            "streaks": [
                {"p0": s.p0, "p1": s.p1, "strength": s.strength} for s in self.streaks
            ],
            "meta": self.meta,
        }
        return json.dumps(payload, indent=2)

# =========================
# Pipeline Controller / IO
# =========================

class PipelineController:
    def __init__(self):
        self.pipeline = None

    @staticmethod
    def load_fits_as_image(path: str) -> Tuple[ReducedImage, ImageMetadata]:
        if fits is None:
            raise RuntimeError("astropy is required to read FITS: pip install astropy")
        with fits.open(path, memmap=False) as hdul:
            hdu = None
            for cand in hdul:
                if getattr(cand, 'data', None) is not None:
                    hdu = cand
                    break
            if hdu is None:
                raise RuntimeError("No image/table HDU found in FITS")
            data = np.array(hdu.data, dtype=float)
            hdr = dict(hdu.header)
            if data.ndim == 1:
                img = ReducedImage(data, meta=hdr)
            else:
                img = ReducedImage(data.squeeze(), meta=hdr)
            return img, ImageMetadata(hdr)

    def processOne(self, path: str, out_prefix: Optional[str] = None) -> Result:
        img, meta = self.load_fits_as_image(path)
        sampler = ProfileSampler(img)
        streakDet = StreakDetector(sampler)
        bg = BackgroundEstimator()

        streaks = streakDet.detect(img, row_step=max(1, img.shape()[0] // 32))
        feats = FeatureExtractor(sampler).extract(img, streaks, bg)
        clf = RuleBasedClassifier()
        label = clf.classify(feats)
        streak_present = clf.streak_present_bool(label)

        result = Result(
            label=label,
            streak_present=streak_present,
            features=feats,
            streaks=streaks,
            meta={"path": path},
        )

        if out_prefix:
            with open(f"{out_prefix}_summary.json", "w") as f:
                f.write(result.toJSON())
            self._save_peaks_csv(img, sampler, out_prefix)
            self._save_plot(img, result, out_prefix)
        return result

    def processBatch(self, dir_glob: str):
        import glob
        results: List[Result] = []
        for path in glob.glob(dir_glob):
            results.append(self.processOne(path, out_prefix=None))
        return results

    def _save_peaks_csv(self, img: ReducedImage, sampler: ProfileSampler, prefix: str):
        rows = list(range(0, img.shape()[0], max(1, img.shape()[0] // 16)))
        with open(f"{prefix}_peaks.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["axis", "index", "x", "y_s", "snr", "fwhm"])
            for y in rows:
                prof = sampler.row(y)
                pk = PeakDetector().find(prof, minProminence=6.0, minDistance=2.0)
                for p in pk:
                    w.writerow(["row", y, p.x, p.y, p.snr, p.fwhm])

    def _save_plot(self, img: ReducedImage, result: Result, prefix: str):
        arr = img.toArray()
        H, W = arr.shape
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        im = ax.imshow(arr, cmap="gray", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="intensity (a.u.)")
        for s in result.streaks:
            (x0, y0), (x1, y1) = s.p0, s.p1
            ax.plot([x0, x1], [y0, y1], lw=1.5, color="tab:red", alpha=0.8)
        ax.set_title(result.summary())
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        plt.tight_layout()
        plt.savefig(f"{prefix}_plot.png", dpi=160)
        plt.close(fig)

# =========================
# CLI
# =========================

def _build_argparser():
    ap = argparse.ArgumentParser(description="Star Streak Detector – SG profiles + peaks + derivative voting")
    ap.add_argument("--fits", dest="fits_path", required=True, help="Path to FITS reduced image or 1D spectrum")
    ap.add_argument("--out", dest="out_prefix", default="streak", help="Output prefix for report files")
    return ap

def main():
    ap = _build_argparser()
    args = ap.parse_args()
    ctl = PipelineController()
    res = ctl.processOne(args.fits_path, out_prefix=args.out_prefix)
    # Clear, human-friendly verdict:
    print(res.summary())
    # Full JSON to stdout (also written to <out>_summary.json if --out provided)
    print(res.toJSON())

if __name__ == "__main__":
    main()
