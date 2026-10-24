"""Gaussian-calibration bias study for CTR-image bunch-length monitors.

A monitor calibrated with Gaussian bunches (image width -> sigma_t) is applied to realistic,
non-Gaussian compressed bunches from ASTRA. Same forward model throughout:
tracked particles -> coherent CTR on the foil -> band-limited ASM -> broadband (0.1-3 THz) screen images.
"""
import os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.special import ndtri
from scipy.optimize import minimize_scalar, least_squares
import beamline as b, ctr, run_scan, make_gif as mg

OUT = os.path.join(b.HERE, "results", "calib")
ZS = np.array([0.1, 0.3, 0.5, 1.0, 2.0])          # screen distances from foil [m]
SIG_CAL = np.geomspace(8e-15, 1.5e-12, 36)         # Gaussian calibration grid [s]
C = 299792458.0


def images(x, y, t):
    Ex, Ey = ctr.bunch_field(mg.G, mg.F_THZ * 1e12, x, y, t, foil_radius=mg.FOIL_R)
    out = []
    for z in ZS:
        Px, Py = mg.propagate_both(Ex, Ey, z)
        out.append(ctr.fluence(Px, Py).sum(0) * mg.DF)
    return np.array(out)


def widths(img):
    """Image-size metrics per screen: outer FWHM of the horizontal and vertical cuts (as used by
    CTR-imaging monitors) and rms radius over the screen."""
    X, Y, x = mg.G.X, mg.G.Y, mg.G.x
    c = mg.G.n // 2
    res = []
    for im in img:
        prof = 0.5 * (im[c] + im[:, c])
        h = 0.5 * prof.max()
        idx = np.nonzero(prof >= h)[0]
        i0, i1 = idx[0], idx[-1]
        xl = np.interp(h, [prof[i0 - 1], prof[i0]], [x[i0 - 1], x[i0]]) if i0 > 0 else x[0]
        xr = np.interp(h, [prof[i1 + 1], prof[i1]], [x[i1 + 1], x[i1]]) if i1 < len(x) - 1 else x[-1]
        rr = np.sqrt((im * (X**2 + Y**2)).sum() / im.sum())
        res.append((xr - xl, rr))
    return np.array(res)          # (nz, 2): [fwhm, rms radius]


def reference_transverse():
    d = np.loadtxt(os.path.join(b.RUNDIR, "h000p000.0345.001"))
    x, y, _, _ = mg.bunch_at_foil(d)
    return x, y


def gaussian_case(sig):
    f = os.path.join(OUT, f"gauss_{sig*1e15:08.2f}fs.npz")
    if not os.path.exists(f):
        x, y = reference_transverse()
        rng = np.random.default_rng(7)
        t = sig * ndtri((np.arange(len(x)) + 0.5) / len(x))      # stratified Gaussian, no shot noise
        rng.shuffle(t)
        np.savez(f, w=widths(images(x, y, t)), sig=sig)
    return dict(np.load(f))


def astra_case(h):
    tag = f"h{h:07.3f}".replace(".", "p")
    f = os.path.join(OUT, f"{tag}.npz")
    if not os.path.exists(f):
        d = np.loadtxt(os.path.join(b.RUNDIR, f"{tag}.{int(round(b.Z_STOP*100)):04d}.001"))
        x, y, t, de = mg.bunch_at_foil(d)
        np.savez(f, w=widths(images(x, y, t)), **true_lengths(t), s=h / run_scan.H_FULL, h=h)
    return dict(np.load(f))


def true_lengths(t):
    """Different 'true' bunch-length definitions for the same bunch."""
    t = t - np.median(t)
    rms = t.std()
    # FWHM of the current profile (fine histogram, light smoothing), expressed as Gaussian-equivalent sigma
    w = max(rms * 8, 50e-15)
    hist, e = np.histogram(t, bins=2000, range=(-w, w))
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0)**2); hist = np.convolve(hist, k / k.sum(), "same")
    tc = 0.5 * (e[1:] + e[:-1]); above = tc[hist >= hist.max() / 2]
    fwhm = above[-1] - above[0]
    # sigma of the Gaussian whose |F|^2 best matches this bunch's |F|^2 over the detected band
    ff = ctr.form_factor(t, mg.F_THZ * 1e12)
    om = 2 * np.pi * mg.F_THZ * 1e12
    fit = minimize_scalar(lambda s_fs: np.sum((np.exp(-(om * s_fs * 1e-15)**2) - ff)**2), bounds=(1.0, 3000.0),
                          method="bounded", options=dict(xatol=1e-3))
    q = np.percentile(t, [15.87, 84.13])
    return dict(rms=rms, fwhm_sig=fwhm / 2.3548, ff_sig=fit.x * 1e-15, core68=(q[1] - q[0]) / 2)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor(6) as ex:
        G = list(ex.map(gaussian_case, SIG_CAL)); print("calibration done", flush=True)
        A = list(ex.map(astra_case, run_scan.H_LIST)); print("astra done", flush=True)
