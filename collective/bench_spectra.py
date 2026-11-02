"""ASTRA vs Ocelot benchmark of the chicane scan, carried through to the CTR spectra.

ASTRA  : make_gif.compute (results/h*.npz, needs the ASTRA runs of run_scan.py)
Ocelot : ctr_gif.compute  (collective/results/<tag>_<mode>_ctr.npz, needs collective/track.py none,both)
Same foil (40 mm, 0.30 m after B4), same ASM grid; spectrum = energy per Hz on the 128 mm screen 0.5 m from the foil.
Writes media/collective/bench_spectra.pdf and bench_spectra_numbers.txt.

usage (from the repository root):  python collective/bench_spectra.py
"""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from track import b, rs
import make_gif as mg
from ctr_gif import compute as ocelot_ctr

MEDIA = os.path.join(b.HERE, "media", "collective")
F = mg.F_THZ
S = rs.H_LIST / rs.H_FULL
FLOOR = 20 / b.NMACRO                          # mask |F|^2 below the macro-particle shot-noise floor
CASES = (0.0, 0.525, 0.94, 1.0, 1.06)          # hR56 values shown as spectra
LAB = {"none": "no collective effects", "both": "CSR + space charge"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "savefig.bbox": "tight", "savefig.dpi": 300})


def masked(r):
    return np.where(r["ff"] > FLOOR, r["spec"] * 1e21, np.nan)       # J/Hz -> nJ/THz


def energy(rs_):
    return np.array([r["screens"][1].sum() * mg.G.dx**2 for r in rs_]) * 1e9   # nJ on the 0.5 m screen


if __name__ == "__main__":
    A = [mg.compute(h) for h in rs.H_LIST]
    O = {m: [ocelot_ctr((h, m)) for h in rs.H_LIST] for m in ("none", "both")}
    ks = [int(np.argmin(abs(S - s))) for s in CASES]
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(ks)))

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4), height_ratios=(1.6, 1))
    for k, c in zip(ks, cols):
        ax[0, 0].semilogy(F, masked(A[k]), "-", color=c, lw=1.2, label=f"$hR_{{56}}$={S[k]:.2f}")
        ax[0, 0].semilogy(F, masked(O["none"][k]), "o", color=c, ms=2.5, mfc="none", mew=0.7)
        ax[0, 0].semilogy(F, masked(O["both"][k]), "x", color=c, ms=2.5, mew=0.7)
        ax[1, 0].plot(F, masked(O["none"][k]) / masked(A[k]), "o-", color=c, ms=2, lw=0.6)
        ax[1, 0].plot(F, masked(O["both"][k]) / masked(A[k]), "x--", color=c, ms=2.5, lw=0.6)
    ax[0, 0].set_ylabel(r"d$W$/d$\nu$, 0.5 m screen [nJ/THz]")
    ax[0, 0].set_title("line: ASTRA   o: Ocelot, none   x: Ocelot, CSR+SC", fontsize=7)
    ax[0, 0].legend(fontsize=6, frameon=False, ncol=2)
    ax[1, 0].axhline(1, color="k", lw=0.5); ax[1, 0].set_ylim(0.5, 1.5)
    ax[1, 0].set_ylabel("Ocelot / ASTRA"); ax[1, 0].set_xlabel("frequency [THz]")

    EA, EN, EB = energy(A), energy(O["none"]), energy(O["both"])
    ax[0, 1].semilogy(S, EA, "k-", lw=1.2, label="ASTRA")
    ax[0, 1].semilogy(S, EN, "o", color="C0", ms=3, mfc="none", label="Ocelot, " + LAB["none"])
    ax[0, 1].semilogy(S, EB, "x", color="C3", ms=3, label="Ocelot, " + LAB["both"])
    ax[0, 1].set_ylabel("CTR energy, 0.5 m screen [nJ]"); ax[0, 1].legend(fontsize=6, frameon=False)
    ax[1, 1].plot(S, EN / EA, "o-", color="C0", ms=2.5, lw=0.7)
    ax[1, 1].plot(S, EB / EA, "x-", color="C3", ms=3, lw=0.7)
    ax[1, 1].axhline(1, color="k", lw=0.5)
    ax[1, 1].set_ylabel("Ocelot / ASTRA"); ax[1, 1].set_xlabel(r"$hR_{56}$")
    fig.tight_layout()
    os.makedirs(MEDIA, exist_ok=True)
    fig.savefig(os.path.join(MEDIA, "bench_spectra.pdf")); plt.close(fig)

    out = os.path.join(MEDIA, "bench_spectra_numbers.txt")
    with open(out, "w") as fh:
        fh.write("ASTRA vs Ocelot (no collective effects / CSR+SC), CTR on the 0.5 m screen\n")
        fh.write(" hR56  sig_t[fs] A / none / both     E[nJ] A / none / both    none/A  both/A   "
                 "max|spec none/A-1| over unmasked f\n")
        for i, s in enumerate(S):
            r = masked(O["none"][i]) / masked(A[i])
            fh.write(f" {s:5.3f}  {A[i]['sig_t']*1e15:7.2f} {O['none'][i]['sig_t']*1e15:7.2f} {O['both'][i]['sig_t']*1e15:7.2f}"
                     f"   {EA[i]:8.3f} {EN[i]:8.3f} {EB[i]:8.3f}   {EN[i]/EA[i]:6.3f}  {EB[i]/EA[i]:6.3f}   "
                     f"{np.nanmax(abs(r - 1)):.3f}\n")
        fh.write(f"\nmax energy: ASTRA {EA.max():.2f} nJ at {S[EA.argmax()]:.3f}, Ocelot none {EN.max():.2f} at "
                 f"{S[EN.argmax()]:.3f}, CSR+SC {EB.max():.2f} at {S[EB.argmax()]:.3f}\n")
    print(open(out).read())
