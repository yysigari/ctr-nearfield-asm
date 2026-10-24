"""Bunch compression -> coherent transition radiation in the near field, as GIFs.

Pipeline per chirp setting:  ASTRA chicane tracking -> ballistic drift to foil -> coherent CTR source field
(Ginzburg-Frank kernel convolved with the complex macro-particle density) -> band-limited ASM to screens.
"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import imageio.v2 as imageio
import beamline as b, ctr, asm, run_scan

OUT = os.path.join(b.HERE, "results"); os.makedirs(OUT, exist_ok=True)
FOIL_DRIFT = 0.30 - (b.Z_STOP - b.Z_EXIT)      # foil sits 0.30 m after the last dipole
FOIL_R = 0.020
F_THZ = np.linspace(0.1, 3.0, 30)
DF = (F_THZ[1] - F_THZ[0]) * 1e12
SCREENS = [0.1, 0.5, 2.0]                       # m downstream of the foil
G = ctr.Grid(256, 0.064)
NEAR = lambda z, f: z / (b.GAMMA**2 * ctr.C / (f * 1e12))   # z / (gamma^2 lambda)


def bunch_at_foil(d):
    x, y, t, de = b.bunch_coordinates(d)
    pz = d[:, 5].copy(); pz[1:] += d[0, 5]
    good = d[:, 9] >= 0; good[0] = True
    xp, yp = d[good, 3] / pz[good], d[good, 4] / pz[good]
    return x + xp * FOIL_DRIFT, y + yp * FOIL_DRIFT, t, de


def propagate_both(Ex, Ey, z):
    f = F_THZ * 1e12
    return asm.propagate(Ex, G.dx, f, z, pad=3), asm.propagate(Ey, G.dx, f, z, pad=3)


def compute(h):
    tag = f"h{h:07.3f}".replace(".", "p")
    cache = os.path.join(OUT, f"{tag}.npz")
    if os.path.exists(cache):
        return dict(np.load(cache))
    d = np.loadtxt(os.path.join(b.RUNDIR, f"{tag}.{int(round(b.Z_STOP*100)):04d}.001"))
    x, y, t, de = bunch_at_foil(d)
    Ex, Ey = ctr.bunch_field(G, F_THZ * 1e12, x, y, t, foil_radius=FOIL_R)
    screens, spec = [], None
    for z in SCREENS:
        Px, Py = propagate_both(Ex, Ey, z)
        Fl = ctr.fluence(Px, Py)                         # J/Hz/m^2 per frequency
        screens.append(Fl.sum(0) * DF)                   # broadband fluence J/m^2
        if z == 0.5:
            spec = Fl.sum((1, 2)) * G.dx**2              # J/Hz collected by the 128 mm screen
    H, te, dee = np.histogram2d(t * 1e15, de * 1e2, bins=(160, 120), range=[[-200, 200], [-1.6, 1.6]])
    tt = t - t.mean()
    cur, ce = np.histogram(tt * 1e15, bins=400, range=(-2500, 2500))
    res = dict(h=h, s=h / run_scan.H_FULL, sig_t=t.std(), screens=np.array(screens), spec=spec,
               ff=ctr.form_factor(t, F_THZ * 1e12), lps=H, cur=cur * (b.Q_PC * 1e-12 / len(t)) / (12.5e-15),
               cur_t=0.5 * (ce[1:] + ce[:-1]), t=tt[::5].astype(np.float32), de=de[::5].astype(np.float32))
    np.savez(cache, **res)
    return res


def frame(r, allr, vmax, path):
    from matplotlib.colors import PowerNorm
    fig = plt.figure(figsize=(16.5, 9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.1], hspace=0.42, wspace=0.42,
                  left=0.05, right=0.95, top=0.88, bottom=0.07)
    # (a) longitudinal phase space + current
    a = fig.add_subplot(gs[0, 0])
    tf, df = r["t"] * 1e15, r["de"] * 1e2
    a.hist2d(tf, df, bins=(220, 140), range=[[-1800, 1800], [-1.5, 1.5]], cmap="Blues", cmin=1)
    a.set_xlabel("t − ⟨t⟩ [fs]   (head at left)"); a.set_ylabel("ΔE/E [%]")
    a2 = a.twinx(); a2.plot(r["cur_t"], np.maximum(r["cur"], 1e-3), color="C3", lw=1.2)
    a2.set_yscale("log"); a2.set_ylim(1, max(x["cur"].max() for x in allr) * 1.5)
    a2.set_ylabel("current [A]", color="C3"); a2.tick_params(axis="y", colors="C3")
    a.set_title(f"ASTRA: bunch at the foil    σ_t = {r['sig_t']*1e15:.0f} fs,  I_peak ≈ {r['cur'].max():.0f} A", fontsize=10.5)
    ins = a.inset_axes([0.03, 0.56, 0.36, 0.4])
    w = max(4 * r["sig_t"] * 1e15, 60)
    ins.hist2d(tf, df, bins=80, range=[[-w, w], [-1.5, 1.5]], cmap="Blues", cmin=1)
    ins.set_xticks([-w / 2, 0, w / 2]); ins.set_xticklabels([f"{-w/2:.0f}", "0", f"{w/2:.0f} fs"], fontsize=7)
    ins.set_yticks([]); ins.set_title("zoom", fontsize=7, pad=2)
    # (b) spectrum (masked below the macro-particle shot-noise floor |F|^2 ~ 1/N_macro)
    a = fig.add_subplot(gs[0, 1])
    spec = np.where(r["ff"] > 20 / b.NMACRO, r["spec"] * 1e21, np.nan)        # J/Hz -> nJ/THz
    a.semilogy(F_THZ, spec, "C1", lw=2)
    a.set_ylim(1e-2, max(x["spec"].max() for x in allr) * 1e21 * 3); a.set_xlim(0, 3)
    a.set_xlabel("frequency [THz]"); a.set_ylabel("dW/dν on screen @0.5 m [nJ/THz]", color="C1")
    a3 = a.twinx(); a3.plot(F_THZ, r["ff"], "k--", lw=1); a3.set_ylim(0, 1.05); a3.set_ylabel("form factor |F(ν)|²  (dashed)")
    a.set_title("coherent CTR spectrum  ∝ N_e² |F(ν)|² × single-electron TR", fontsize=10.5)
    # (c) energy vs compression
    a = fig.add_subplot(gs[0, 2])
    s = np.array([x["s"] for x in allr]); E = np.array([x["screens"][1].sum() * G.dx**2 for x in allr]) * 1e9
    a.plot(s, E, "-", color="0.6"); a.plot(r["s"], r["screens"][1].sum() * G.dx**2 * 1e9, "o", color="C3", ms=9)
    a.set_xlabel("compression setting  h·R₅₆   (1 = full compression)"); a.set_ylabel("CTR energy on screen @0.5 m [nJ]")
    a.axvline(1, color="k", lw=0.5, ls=":"); a.set_title("under → full → over-compression", fontsize=10.5)
    # (d) screens, fixed absolute scale across frames (power-law colour map to keep dim frames visible)
    zoom = {0.1: 25, 0.5: 50, 2.0: 64}
    for j, z in enumerate(SCREENS):
        a = fig.add_subplot(gs[1, j])
        ext = [G.x[0] * 1e3, G.x[-1] * 1e3] * 2
        im = a.imshow(r["screens"][j] * 1e6, extent=ext, origin="lower", cmap="inferno",
                      norm=PowerNorm(0.45, vmin=0, vmax=vmax[j] * 1e6))
        a.add_patch(plt.Circle((0, 0), FOIL_R * 1e3, fill=False, ls=":", color="w", lw=0.8))
        a.set_xlim(-zoom[z], zoom[z]); a.set_ylim(-zoom[z], zoom[z])
        a.set_title(f"screen {z} m from foil  —  z/γ²λ = {NEAR(z, 0.3):.2f} (0.3 THz) … {NEAR(z, 3.0):.1f} (3 THz)\n"
                    f"peak {r['screens'][j].max()*1e6:.3g} pJ/mm²", fontsize=10)
        a.set_xlabel("x [mm]")
        if j == 0: a.set_ylabel("y [mm]")
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03); cb.set_label("fluence 0.1–3 THz [pJ/mm²]")
    fig.suptitle(f"30 MeV, {b.Q_PC:.0f} pC bunch compressed in a 4-dipole chicane (ASTRA, R₅₆ = {run_scan.R56*1e3:.1f} mm)  →  "
                 f"coherent transition radiation from a Ø{FOIL_R*2e3:.0f} mm foil  →  band-limited angular-spectrum propagation\n"
                 "dotted circle = foil edge;  colour scale fixed across frames", fontsize=12)
    fig.savefig(path, dpi=72); plt.close(fig)


def propagation_gif(r_h):
    """Fully compressed bunch: sweep the screen distance through the near field."""
    tag = f"h{r_h:07.3f}".replace(".", "p")
    d = np.loadtxt(os.path.join(b.RUNDIR, f"{tag}.{int(round(b.Z_STOP*100)):04d}.001"))
    x, y, t, _ = bunch_at_foil(d)
    Ex, Ey = ctr.bunch_field(G, F_THZ * 1e12, x, y, t, foil_radius=FOIL_R)
    zs = np.geomspace(0.01, 3.0, 48)
    sel = {0.5: np.argmin(abs(F_THZ - 0.5)), 2.0: np.argmin(abs(F_THZ - 2.0))}
    rows, imgs = [], []
    for z in zs:
        Px, Py = propagate_both(Ex, Ey, z)
        Fl = ctr.fluence(Px, Py)
        bb = Fl.sum(0) * DF
        imgs.append((bb, Fl[sel[0.5]], Fl[sel[2.0]]))
        rows.append(bb[G.n // 2] / bb[G.n // 2].max())
    frames = []
    ext = [G.x[0] * 1e3, G.x[-1] * 1e3] * 2
    for i, z in enumerate(zs):
        fig, ax = plt.subplots(1, 4, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25]))
        for a, im, ttl in zip(ax[:3], imgs[i], ["broadband 0.1–3 THz", "0.5 THz", "2.0 THz"]):
            a.imshow(im / im.max(), extent=ext, origin="lower", cmap="inferno", vmin=0, vmax=1)
            a.add_patch(plt.Circle((0, 0), FOIL_R * 1e3, fill=False, ls=":", color="w", lw=0.8))
            f = {"0.5 THz": 0.5, "2.0 THz": 2.0}.get(ttl)
            a.set_title(ttl + (f"   z/γ²λ = {NEAR(z, f):.2f}" if f else "") + "\n(normalised)", fontsize=10)
            a.set_xlabel("x [mm]")
        ax[0].set_ylabel("y [mm]")
        M = np.full((len(zs), G.n), np.nan); M[: i + 1] = rows[: i + 1]
        ax[3].pcolormesh(G.x * 1e3, zs, M, cmap="inferno", vmin=0, vmax=1, shading="auto")
        ax[3].axhline(z, color="c", lw=1); ax[3].set_yscale("log"); ax[3].set_ylim(zs[0], zs[-1])
        ax[3].set_xlabel("x [mm]  (y = 0 cut)"); ax[3].set_ylabel("distance from foil z [m]")
        ax[3].set_title("broadband profile vs distance (each row normalised)", fontsize=10)
        fig.suptitle(f"Fully compressed bunch (σ_t = {t.std()*1e15:.0f} fs): CTR near field → far field,  z = {z:.3f} m", fontsize=12)
        fig.tight_layout()
        p = os.path.join(OUT, "prop_frame.png"); fig.savefig(p, dpi=72); plt.close(fig)
        frames.append(imageio.imread(p))
    imageio.mimsave(os.path.join(b.HERE, "media", "ctr_near_to_far_field.gif"), frames, duration=0.18, loop=0)


if __name__ == "__main__":
    hs = [h for h in run_scan.H_LIST
          if os.path.exists(os.path.join(b.RUNDIR, f"h{h:07.3f}".replace(".", "p") + f".{int(round(b.Z_STOP*100)):04d}.001"))]
    allr = []
    for h in hs:
        allr.append(compute(h)); print(f"h={h:.2f} sig_t={allr[-1]['sig_t']*1e15:.0f} fs", flush=True)
    vmax = [max(r["screens"][j].max() for r in allr) for j in range(len(SCREENS))]
    frames = []
    for i, r in enumerate(allr):
        p = os.path.join(OUT, f"frame_{i:02d}.png"); frame(r, allr, vmax, p); frames.append(imageio.imread(p))
    seq = frames + [frames[-1]] * 3 + frames[::-1] + [frames[0]] * 3            # ping-pong loop
    imageio.mimsave(os.path.join(b.HERE, "media", "bunch_compression_ctr_nearfield.gif"), seq, duration=0.22, loop=0)
    if "--prop" in sys.argv:
        best = min(allr, key=lambda r: r["sig_t"])
        propagation_gif(float(best["h"]))
