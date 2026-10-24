"""Analyse the Gaussian-calibration bias study (run calibration_bias.py first)."""
import os, glob
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import calibration_bias as cb, run_scan

FIG = os.path.join(cb.b.HERE, "media", "calibration"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "savefig.bbox": "tight", "savefig.dpi": 300})
G = [cb.gaussian_case(s) for s in cb.SIG_CAL]
A = [cb.astra_case(h) for h in run_scan.H_LIST]
for a in A:
    tag = f"h{float(a['h']):07.3f}".replace(".", "p")
    d = np.loadtxt(os.path.join(cb.b.RUNDIR, f"{tag}.{int(round(cb.b.Z_STOP*100)):04d}.001"))
    a.update(cb.true_lengths(cb.mg.bunch_at_foil(d)[2]))
sig = np.array([g["sig"] for g in G]); W = np.array([g["w"] for g in G])      # (nsig, nz, 2)
ls = np.log(sig)
M = {0: "FWHM", 1: "rms radius"}


def invert(wmeas, iz, m):
    """sigma from one screen via the Gaussian calibration curve (log-sigma interpolation)."""
    curve = W[:, iz, m]
    j = np.argmin(abs(curve - wmeas))
    grid = np.linspace(ls[0], ls[-1], 4000)
    c = np.interp(grid, ls, curve)
    k = np.argmin(abs(c - wmeas))
    sat = k in (0, len(grid) - 1)
    return np.exp(grid[k]), sat


def joint(wmeas, m):
    """single Gaussian sigma fitted to all screens; returns sigma and rms relative residual"""
    use = [iz for iz in range(len(cb.ZS)) if W[0, iz, m] / W[-1, iz, m] < 0.98]
    def cost(lsig):
        model = np.array([np.interp(lsig, ls, W[:, iz, m]) for iz in use])
        return np.mean((model / wmeas[use, m] - 1)**2)
    r = minimize_scalar(cost, bounds=(ls[0], ls[-1]), method="bounded")
    return np.exp(r.x), np.sqrt(r.fun)


s = np.array([a["s"] for a in A]); order = np.argsort(s)
true = {k: np.array([a[k] for a in A]) for k in ["rms", "fwhm_sig", "ff_sig", "core68"]}
rows = []
for a in A:
    single = {(iz, m): invert(a["w"][iz, m], iz, m) for iz in range(len(cb.ZS)) for m in (0, 1)}
    rows.append(dict(single=single, joint={m: joint(a["w"], m) for m in (0, 1)}))

# ---------- figure: calibration curves with ASTRA bunches placed at their true rms ----------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
for m, a_ in zip((0, 1), ax):
    for iz, z in enumerate(cb.ZS):
        l, = a_.semilogx(sig * 1e15, W[:, iz, m] * 1e3, lw=0.9, label=f"z = {z} m")
        a_.semilogx(true["rms"] * 1e15, np.array([a["w"][iz, m] for a in A]) * 1e3, "o", ms=2.2, color=l.get_color())
    a_.set_xlabel(r"$\sigma_t$ [fs]")
    a_.set_ylabel(f"image {M[m]} [mm]")
ax[0].legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "calibration_curves.pdf")); plt.close(fig)

# ---------- figure: inferred / true ----------
iz05 = list(cb.ZS).index(0.5)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
for m, a_ in zip((0, 1), ax):
    inf = np.array([r["single"][(iz05, m)][0] for r in rows])
    sat = np.array([r["single"][(iz05, m)][1] for r in rows])
    for k, lab, st in [("rms", r"true rms $\sigma_t$", "o-"), ("fwhm_sig", r"FWHM/2.355", "s-"), ("ff_sig", r"$\sigma$ of best-fit Gaussian $|F|^2$", "^-")]:
        a_.plot(s[order], (inf / true[k])[order], st, ms=2.5, lw=0.8, label=lab)
    a_.plot(s[sat], (inf / true["rms"])[sat], "kx", ms=5, label="calibration saturated")
    a_.axhline(1, color="k", lw=0.5); a_.set_ylim(0.85, 2.1)
    a_.set_xlabel(r"$hR_{56}$"); a_.set_ylabel(r"$\sigma_{\rm inferred}$ / reference")
    a_.set_title(f"Gaussian calibration of image {M[m]}, screen at 0.5 m", fontsize=8)
ax[0].legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "calibration_bias.pdf")); plt.close(fig)

# ---------- figure: multi-screen consistency ----------
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for m, st in ((0, "o-"), (1, "s-")):
    res = np.array([r["joint"][m][1] for r in rows])
    ax.semilogy(s[order], res[order] * 100, st, ms=2.5, lw=0.8, label=f"image {M[m]}")
ax.set_xlabel(r"$hR_{56}$"); ax.set_ylabel("rms misfit of joint Gaussian fit [%]")
ax.legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "calibration_misfit.pdf")); plt.close(fig)

# ---------- numbers ----------
with open(os.path.join(FIG, "calibration_numbers.txt"), "w") as fh:
    fh.write("monotonically increasing calibration (FWHM, rms) per z: " + str([[bool(np.all(np.diff(W[:, iz, m]) >= 0)) for m in (0, 1)] for iz in range(len(cb.ZS))]) + "\n")
    fh.write("relative change of width over sigma 8->1500 fs per z (FWHM, rms): " +
             str([[round(W[0, iz, m] / W[-1, iz, m], 3) for m in (0, 1)] for iz in range(len(cb.ZS))]) + "\n")
    fh.write(f"{'s':>5} {'rms':>7} {'fwhm/2.35':>9} {'ffsig':>7} {'core68':>7} | " +
             " ".join(f"FW@{z}" for z in cb.ZS) + " | " + " ".join(f"RR@{z}" for z in cb.ZS) + " | jointFW jointRR misFW% misRR%\n")
    for i in order:
        r = rows[i]
        fh.write(f"{s[i]:5.2f} {true['rms'][i]*1e15:7.1f} {true['fwhm_sig'][i]*1e15:9.1f} {true['ff_sig'][i]*1e15:7.1f} {true['core68'][i]*1e15:7.1f} | " +
                 " ".join(f"{r['single'][(iz,0)][0]*1e15:6.0f}{'*' if r['single'][(iz,0)][1] else ' '}" for iz in range(len(cb.ZS))) + " | " +
                 " ".join(f"{r['single'][(iz,1)][0]*1e15:6.0f}{'*' if r['single'][(iz,1)][1] else ' '}" for iz in range(len(cb.ZS))) + " | " +
                 f"{r['joint'][0][0]*1e15:6.0f} {r['joint'][1][0]*1e15:6.0f} {r['joint'][0][1]*100:5.1f} {r['joint'][1][1]*100:5.1f}\n")
print(open(os.path.join(FIG, "calibration_numbers.txt")).read())

with open(os.path.join(FIG, "calibration_numbers.txt"), "a") as fh:
    fh.write("\nlog-sensitivity dlnW/dln(sigma) of the Gaussian calibration (FWHM | rms radius):\n")
    for st in (10, 20, 50, 100, 300, 1000):
        k = np.argmin(abs(sig - st * 1e-15))
        sl = lambda iz, m: np.gradient(np.log(W[:, iz, m]), ls)[k]
        fh.write(f"  sigma={st:5d} fs: " + "  ".join(f"z={z}: {sl(iz,0):+.3f} | {sl(iz,1):+.3f}" for iz, z in enumerate(cb.ZS)) + "\n")


# ---------- figure: sensitivity (log-derivative) and required width precision ----------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
fine = np.linspace(ls[0], ls[-1], 400)
for m, a_ in zip((0, 1), ax):
    for iz, z in enumerate(cb.ZS):
        if W[0, iz, m] / W[-1, iz, m] > 0.98:
            continue                                   # clipped by the screen: no sensitivity
        sl = np.gradient(np.log(W[:, iz, m]), ls)
        a_.loglog(sig * 1e15, np.clip(sl, 1e-4, None), lw=0.9, label=f"z = {z} m")
    a_.set_xlabel(r"Gaussian $\sigma_t$ [fs]"); a_.set_ylabel(r"sensitivity $d\ln W/d\ln\sigma_t$")
    a_.set_title(f"image {M[m]}", fontsize=8); a_.set_ylim(1e-3, 1)
    a2 = a_.secondary_yaxis("right", functions=(lambda y: 10 * y, lambda y: y / 10))
    a2.set_ylabel(r"width precision for 10 % $\sigma_t$ [%]")
ax[0].legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "calibration_sensitivity.pdf")); plt.close(fig)

