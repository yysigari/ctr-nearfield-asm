"""Quick start without ASTRA: load a shipped ASTRA distribution, build the coherent CTR field on the foil
and propagate it with the band-limited ASM to a screen.

    python example_from_distribution.py distributions/h028p281.0345.001.gz --z 0.5
"""
import argparse, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ctr, make_gif as mg

p = argparse.ArgumentParser()
p.add_argument("dist", help="ASTRA phase-space file at the chicane exit (.gz allowed)")
p.add_argument("--z", type=float, default=0.5, help="screen distance from the foil [m]")
a = p.parse_args()

x, y, t, de = mg.bunch_at_foil(np.loadtxt(a.dist))
Ex, Ey = ctr.bunch_field(mg.G, mg.F_THZ * 1e12, x, y, t, foil_radius=mg.FOIL_R)
Px, Py = mg.propagate_both(Ex, Ey, a.z)
img = ctr.fluence(Px, Py).sum(0) * mg.DF                      # J/m^2, 0.1-3 THz
E_tot = img.sum() * mg.G.dx**2

ext = [mg.G.x[0] * 1e3, mg.G.x[-1] * 1e3] * 2
fig, ax = plt.subplots(figsize=(5, 4.2))
im = ax.imshow(img * 1e6, extent=ext, origin="lower", cmap="inferno")
fig.colorbar(im, label="fluence 0.1–3 THz [pJ/mm²]")
ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
ax.set_title(f"σ_t = {t.std()*1e15:.0f} fs, z = {a.z} m, E = {E_tot*1e9:.1f} nJ", fontsize=10)
out = os.path.splitext(os.path.basename(a.dist.replace(".gz", "")))[0] + f"_z{a.z}m.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"sigma_t = {t.std()*1e15:.1f} fs, energy on screen = {E_tot*1e9:.2f} nJ -> {out}")
