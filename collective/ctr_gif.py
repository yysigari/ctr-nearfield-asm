"""Near-field CTR with and without space charge / CSR, as a GIF over the compression scan (run track.py first).

Same foil, CTR source and band-limited ASM as ../make_gif.py, applied to the Ocelot bunches of the four modes.
Writes media/collective/ctr_collective_effects.gif, ctr_energy.pdf and ctr_numbers.txt.
"""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import PowerNorm
from concurrent.futures import ProcessPoolExecutor
import imageio.v2 as imageio
from track import MODES, OUT, output_file, b, rs
import ctr, make_gif as mg

MEDIA = os.path.join(b.HERE, "media", "collective")
LAB = {"none": "no collective effects", "sc": "space charge", "csr": "CSR", "both": "CSR + space charge"}
COL = {"none": "k", "sc": "C0", "csr": "C3", "both": "C2"}
G = mg.G
TBIN = 2.0                                     # fs, current histogram bin


def compute(args):
    h, mode = args
    f = output_file(h, mode).replace(".ast", "_ctr.npz")
    if not os.path.exists(f):
        x, y, t, _ = mg.bunch_at_foil(np.loadtxt(output_file(h, mode)))
        Ex, Ey = ctr.bunch_field(G, mg.F_THZ * 1e12, x, y, t, foil_radius=mg.FOIL_R)
        screens, spec = [], None
        for z in mg.SCREENS:
            Fl = ctr.fluence(*mg.propagate_both(Ex, Ey, z))     # J/Hz/m^2 per frequency
            screens.append(Fl.sum(0) * mg.DF)                   # broadband fluence J/m^2
            if z == 0.5:
                spec = Fl.sum((1, 2)) * G.dx**2                 # J/Hz on the 128 mm screen
        cur, _ = np.histogram((t - np.median(t)) * 1e15, bins=int(5000 / TBIN), range=(-2500, 2500))
        np.savez(f, screens=np.array(screens, dtype=np.float32), spec=spec, ff=ctr.form_factor(t, mg.F_THZ * 1e12),
                 sig_t=t.std(), cur=cur * (b.Q_PC * 1e-12 / len(t)) / (TBIN * 1e-15))
    return dict(np.load(f))


def frame(k, R, E, vmax, dmax, path, dpi=72):
    S, s = rs.S, rs.S[k]
    fig = plt.figure(figsize=(17, 11.5))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1.15, 0.85], hspace=0.5, wspace=0.35,
                  left=0.05, right=0.96, top=0.89, bottom=0.06)
    tc = np.arange(-2500 + TBIN / 2, 2500, TBIN)
    # (a) current profiles
    a = fig.add_subplot(gs[0, 0])
    w = max(5 * R["none"][k]["sig_t"] * 1e15, 80)
    for m in MODES:
        a.plot(tc, R[m][k]["cur"], color=COL[m], lw=1.1, label=f"{LAB[m]}: σ_t = {R[m][k]['sig_t']*1e15:.1f} fs")
    a.set_xlim(-w, w); a.set_ylim(0, max(R[m][j]["cur"].max() for m in MODES for j in range(len(S))) * 1.45)
    a.set_xlabel("t [fs]   (head at left)"); a.set_ylabel("current [A]")
    a.set_title("bunch current at chicane exit", fontsize=10.5); a.legend(fontsize=7, frameon=False, loc="upper left")
    # (b) spectra (masked below the macro-particle shot-noise floor)
    a = fig.add_subplot(gs[0, 1])
    for m in MODES:
        r = R[m][k]
        a.semilogy(mg.F_THZ, np.where(r["ff"] > 20 / b.NMACRO, r["spec"] * 1e21, np.nan), color=COL[m], lw=1.3)
    a.set_ylim(1e-2, max(R[m][j]["spec"].max() for m in MODES for j in range(len(S))) * 1e21 * 3); a.set_xlim(0, 3)
    a.set_xlabel("frequency [THz]"); a.set_ylabel("dW/dν, screen @0.5 m [nJ/THz]")
    a.set_title("coherent CTR spectrum", fontsize=10.5)
    # (c) energy vs compression, (d) ratio to no collective effects
    a, a2 = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])
    for m in MODES:
        a.plot(S, E[m] * 1e9, color=COL[m], lw=1); a.plot(s, E[m][k] * 1e9, "o", color=COL[m], ms=6)
        a2.plot(S, E[m] / E["none"], color=COL[m], lw=1); a2.plot(s, E[m][k] / E["none"][k], "o", color=COL[m], ms=6)
    a.set_xlabel("h·R₅₆   (1 = full compression)"); a.set_ylabel("CTR energy, screen @0.5 m [nJ]")
    a2.set_xlabel("h·R₅₆"); a2.set_ylabel("energy / no collective effects"); a2.axhline(1, color="0.5", lw=0.5)
    a.set_title("under → full → over-compression", fontsize=10.5)
    a2.set_title("relative change of the CTR energy", fontsize=10.5)
    # (e) screen 0.5 m, one panel per mode, colour scale fixed across modes and frames
    ext = [G.x[0] * 1e3, G.x[-1] * 1e3] * 2
    for j, m in enumerate(MODES):
        a = fig.add_subplot(gs[1, j])
        im = R[m][k]["screens"][1]
        h_ = a.imshow(im * 1e6, extent=ext, origin="lower", cmap="inferno", norm=PowerNorm(0.45, vmin=0, vmax=vmax[1] * 1e6))
        a.add_patch(plt.Circle((0, 0), mg.FOIL_R * 1e3, fill=False, ls=":", color="w", lw=0.8))
        a.set_xlim(-50, 50); a.set_ylim(-50, 50); a.set_xlabel("x [mm]")
        if j == 0:
            a.set_ylabel("y [mm]")
        a.set_title(f"{LAB[m]}, screen 0.5 m\npeak {im.max()*1e6:.3g} pJ/mm²,  energy {E[m][k]*1e9:.1f} nJ",
                    fontsize=10, color=COL[m] if m != "none" else "k")
        fig.colorbar(h_, ax=a, fraction=0.046, pad=0.03).set_label("fluence 0.1–3 THz [pJ/mm²]", fontsize=8)
    # (f) horizontal cuts through the screens at 0.1, 0.5 and 2.0 m
    c = G.n // 2
    zoom = {0.1: 25, 0.5: 50, 2.0: 64}
    for j, z in enumerate(mg.SCREENS):
        a = fig.add_subplot(gs[2, j])
        for m in MODES:
            a.plot(G.x * 1e3, R[m][k]["screens"][j][c] * 1e6, color=COL[m], lw=1.1)
        a.set_xlim(0, zoom[z]); a.set_ylim(0, vmax[j] * 1e6 * 1.05)
        a.set_xlabel("x [mm]  (y = 0)"); a.set_ylabel("fluence [pJ/mm²]")
        a.set_title(f"horizontal cut, screen {z} m", fontsize=10)
    # (g) difference image, CSR + space charge minus none, screen 0.5 m
    a = fig.add_subplot(gs[2, 3])
    dif = (R["both"][k]["screens"][1] - R["none"][k]["screens"][1]) * 1e6
    h_ = a.imshow(dif, extent=ext, origin="lower", cmap="RdBu_r", vmin=-dmax * 1e6, vmax=dmax * 1e6)
    a.add_patch(plt.Circle((0, 0), mg.FOIL_R * 1e3, fill=False, ls=":", color="k", lw=0.8))
    a.set_xlim(-50, 50); a.set_ylim(-50, 50); a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
    a.set_title("(CSR + space charge) − none, screen 0.5 m", fontsize=10)
    fig.colorbar(h_, ax=a, fraction=0.046, pad=0.03).set_label("Δ fluence [pJ/mm²]", fontsize=8)
    fig.suptitle(f"Near-field CTR with and without collective effects  —  h·R₅₆ = {s:.3f}\n"
                 f"30 MeV, {b.Q_PC:.0f} pC bunch tracked with Ocelot (3D space charge, 1D CSR) → Ø{mg.FOIL_R*2e3:.0f} mm foil "
                 "0.30 m after the chicane → band-limited angular-spectrum propagation, 0.1–3 THz;  scales fixed across frames",
                 fontsize=12)
    fig.savefig(path, dpi=dpi); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(MEDIA, exist_ok=True)
    jobs = [(h, m) for m in MODES for h in rs.H_LIST]
    with ProcessPoolExecutor(min(8, os.cpu_count() or 1)) as ex:
        res = list(ex.map(compute, jobs))
    n = len(rs.H_LIST)
    R = {m: res[i * n:(i + 1) * n] for i, m in enumerate(MODES)}
    E = {m: np.array([r["screens"][1].sum() * G.dx**2 for r in R[m]]) for m in MODES}
    vmax = [max(r["screens"][j].max() for m in MODES for r in R[m]) for j in range(len(mg.SCREENS))]
    dmax = max(abs(R["both"][k]["screens"][1] - R["none"][k]["screens"][1]).max() for k in range(len(rs.H_LIST)))

    frames = []
    for k in range(n):
        p = os.path.join(OUT, f"ctr_frame_{k:02d}.png"); frame(k, R, E, vmax, dmax, p); frames.append(imageio.imread(p))
    seq = frames + [frames[-1]] * 3 + frames[::-1] + [frames[0]] * 3            # ping-pong loop
    imageio.mimsave(os.path.join(MEDIA, "ctr_collective_effects.gif"), seq, duration=0.25, loop=0)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for m in MODES:
        ax[0].semilogy(rs.S, E[m] * 1e9, color=COL[m], lw=0.9, label=LAB[m])
        ax[1].plot(rs.S, E[m] / E["none"], color=COL[m], lw=0.9)
    ax[0].set_ylabel("CTR energy, 0.5 m screen [nJ]"); ax[1].set_ylabel("energy / no collective effects")
    for a in ax:
        a.set_xlabel(r"$hR_{56}$")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(MEDIA, "ctr_energy.pdf")); plt.close(fig)

    k1 = int(np.argmin(abs(rs.S - 1.0)))
    with open(os.path.join(MEDIA, "ctr_numbers.txt"), "w") as fh:
        fh.write("CTR energy on the 0.5 m screen [nJ] and ratio to no collective effects\n")
        fh.write("    s   " + "  ".join(f"{m:>8s}" for m in MODES) + "   " + "  ".join(f"{m:>6s}/none" for m in MODES[1:]) + "\n")
        for i, s in enumerate(rs.S):
            fh.write(f" {s:4.2f}  " + "  ".join(f"{E[m][i]*1e9:8.3f}" for m in MODES) + "   " +
                     "  ".join(f"{E[m][i]/E['none'][i]:11.3f}" for m in MODES[1:]) + "\n")
        fh.write("\nmax over scan [nJ]: " + ", ".join(f"{m} {E[m].max()*1e9:.2f} at hR56={rs.S[np.argmax(E[m])]:.3f}" for m in MODES) + "\n")
        fh.write(f"peak fluence at hR56=1 [pJ/mm^2], screens {mg.SCREENS} m: " +
                 "; ".join(f"{m} " + "/".join(f"{R[m][k1]['screens'][j].max()*1e6:.4g}" for j in range(3)) for m in MODES) + "\n")
    print(open(os.path.join(MEDIA, "ctr_numbers.txt")).read())
