"""Checks for the Ocelot model.

1. Lattice: R56 from the Ocelot transfer map, and bunch length versus ASTRA without collective effects
   (uses beamline.run_astra, so existing astra_runs/ from run_scan.py are reused; skipped without ASTRA).
2. Numerical convergence of the space-charge result at full compression (mesh size, kick step).
Writes collective/results/figs/checks.txt.  Use --no-conv to skip part 2.
"""
import os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import track as tr
from track import b, rs
from analysis import beam_metrics

H1 = rs.H_LIST[np.argmin(abs(rs.S - 1.0))]
CONV = {"baseline (63^3, 5 mm)": dict(), "nmesh 95^3": dict(nmesh=95), "step 2.5 mm": dict(step=0.0025)}


def astra_available(h):
    out = os.path.join(b.RUNDIR, f"{tr.tag(h)}.{int(round(b.Z_STOP*100)):04d}.001")
    return os.path.exists(out) or os.path.exists(b.ASTRA)


def conv_job(k):
    return k, beam_metrics(np.loadtxt(tr.track(H1, "sc", out=os.path.join(tr.OUT, "conv", k.split()[0]), **CONV[k])))


if __name__ == "__main__":
    from ocelot import lattice_transfer_map
    lat, _, _ = tr.lattice()
    lines = [f"Ocelot R56 = {lattice_transfer_map(lat, energy=b.P0 * 1e-9)[4, 5] * 1e3:+.3f} mm "
             f"(path length {lat.totalLen:.4f} m)"]
    hs = [rs.H_FULL * s for s in (0.0, 0.84, 1.0, 1.4)]
    if all(astra_available(h) for h in hs):
        lines.append("\nbunch length without collective effects, ASTRA vs Ocelot:")
        for h in hs:
            ta = b.bunch_coordinates(b.run_astra(h))[2]
            to = b.bunch_coordinates(np.loadtxt(tr.track(h, "none")))[2]
            lines.append(f"  hR56={h / rs.H_FULL:4.2f}: ASTRA {ta.std()*1e15:7.1f} fs   Ocelot {to.std()*1e15:7.1f} fs")
    else:
        lines.append(f"\nASTRA not found at {b.ASTRA}; skipping the ASTRA comparison")
    if "--no-conv" not in sys.argv:
        with ProcessPoolExecutor(3) as ex:
            conv = list(ex.map(conv_job, CONV))
        lines.append("\nspace-charge convergence at hR56 = 1.0:")
        for k, r in conv:
            lines.append(f"  {k:22s} rms {r['rms']*1e15:6.2f} fs   eps_x {r['eps_x']*1e6:.3f} um")
    os.makedirs(os.path.join(tr.OUT, "figs"), exist_ok=True)
    open(os.path.join(tr.OUT, "figs", "checks.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
