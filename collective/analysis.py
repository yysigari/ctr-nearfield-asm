"""Beam quantities and figures for the Ocelot scan (run collective/track.py first).

Writes collective/results/figs/scan.pdf, profiles.pdf and numbers.txt.
"""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from track import MODES, OUT, output_file, b, rs

FIG = os.path.join(OUT, "figs")
LAB = {"none": "no collective effects", "sc": "space charge", "csr": "CSR", "both": "CSR + space charge"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "savefig.bbox": "tight", "savefig.dpi": 300})


def fwhm(t, rms):
    """FWHM of the current profile (fine histogram, light smoothing)."""
    w = max(rms * 8, 50e-15)
    hist, e = np.histogram(t, bins=2000, range=(-w, w))
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0)**2)
    hist = np.convolve(hist, k / k.sum(), "same")
    tc = 0.5 * (e[1:] + e[:-1]); above = tc[hist >= hist.max() / 2]
    return above[-1] - above[0]


def coordinates(d):
    """beamline.bunch_coordinates plus x' = px/pz for the alive particles."""
    x, y, t, de = b.bunch_coordinates(d)
    good = d[:, 9] >= 0
    good[0] = True
    pz = d[good, 5] + d[0, 5]           # absolute pz; the reference row is already absolute
    pz[0] = d[0, 5]
    return x, y, t, de, d[good, 3] / pz


def beam_metrics(d):
    x, y, t, de, xp = coordinates(d)
    t = t - np.median(t)
    # projected emittance with the residual linear dispersion removed
    A = np.vstack([de - de.mean(), np.ones_like(de)]).T
    xr = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    xpr = xp - A @ np.linalg.lstsq(A, xp, rcond=None)[0]
    rms = t.std()
    w = max(8 * rms, 50e-15)
    hist, e = np.histogram(t, bins=400, range=(-w, w))
    return dict(rms=rms, fwhm=fwhm(t, rms), sig_d=de.std(), mean_d=de.mean(),
                eps_x=np.sqrt(np.linalg.det(np.cov(xr, xpr))) * b.GAMMA * b.BETA,
                ipk=hist.max() * (b.Q_PC * 1e-12 / len(t)) / (e[1] - e[0]),
                sig_x=x.std(), sig_y=y.std())


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    modes = [m for m in MODES if all(os.path.exists(output_file(h, m)) for h in rs.H_LIST)]
    M = {m: [beam_metrics(np.loadtxt(output_file(h, m))) for h in rs.H_LIST] for m in modes}
    get = lambda m, k: np.array([r[k] for r in M[m]])
    sty = dict(zip(MODES, ("k-", "o-", "s-", "^-")))

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for m in modes:
        kw = dict(ms=2.2, lw=0.8, label=LAB[m])
        ax[0].semilogy(rs.S, get(m, "rms") * 1e15, sty[m], **kw)
        ax[1].plot(rs.S, get(m, "eps_x") * 1e6, sty[m], **kw)
        ax[2].plot(rs.S, -get(m, "mean_d") * b.P0 * 1e-3, sty[m], **kw)
    ax[0].set_ylabel(r"rms $\sigma_t$ [fs]"); ax[1].set_ylabel(r"$\varepsilon_{n,x}$ [mm mrad]")
    ax[2].set_ylabel("mean energy loss [keV]")
    for a_ in ax:
        a_.set_xlabel(r"$hR_{56}$")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(FIG, "scan.pdf")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for m in modes:
        k = np.argmin(get(m, "rms"))
        t = b.bunch_coordinates(np.loadtxt(output_file(rs.H_LIST[k], m)))[2]
        t = t - np.median(t)
        hist, e = np.histogram(t, bins=320, range=(-40e-15, 120e-15))
        ax.plot(0.5 * (e[1:] + e[:-1]) * 1e15, hist * (b.Q_PC * 1e-12 / len(t)) / (e[1] - e[0]), lw=0.8,
                label=f"{LAB[m]} ($hR_{{56}}$={rs.S[k]:.2f}, {get(m, 'rms')[k]*1e15:.1f} fs)")
    ax.set_xlabel("t [fs] (tail positive)"); ax.set_ylabel("current [A]")
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "profiles.pdf")); plt.close(fig)

    out = os.path.join(FIG, "numbers.txt")
    with open(out, "w") as fh:
        for m in modes:
            k = np.argmin(get(m, "rms"))
            fh.write(f"{m:5s}: min rms {get(m,'rms')[k]*1e15:.1f} fs at hR56={rs.S[k]:.3f}, Ipk {get(m,'ipk')[k]:.0f} A, "
                     f"eps_x {get(m,'eps_x')[k]*1e6:.3f} um, max mean loss {-get(m,'mean_d').min()*b.P0*1e-3:.1f} keV\n")
        fh.write("\n    s | rms [fs]: " + " ".join(f"{m:>6s}" for m in modes) + " | eps_x [um]: " +
                 " ".join(f"{m:>6s}" for m in modes) + " | sig_x,y [mm]: " + " ".join(f"{m:>11s}" for m in modes) + "\n")
        for i, s in enumerate(rs.S):
            fh.write(f" {s:4.2f} |           " + " ".join(f"{get(m,'rms')[i]*1e15:6.1f}" for m in modes) + " |             " +
                     " ".join(f"{get(m,'eps_x')[i]*1e6:6.3f}" for m in modes) + " |               " +
                     " ".join(f"{get(m,'sig_x')[i]*1e3:5.3f},{get(m,'sig_y')[i]*1e3:5.3f}" for m in modes) + "\n")
    print(open(out).read())
