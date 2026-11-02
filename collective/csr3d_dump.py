"""Bunches at the middle of each chicane dipole, for the PyCSR3D check (needs Ocelot).

Same lattice and initial bunch as track.py (beamline.make_distribution, same seed), no collective effects. Each dipole is split in half and the
particle array is saved at every mid-dipole point, in Ocelot's coordinates (relative to the reference orbit):
x, y [m], tau [m] (Ocelot convention), p = dE/(p0 c), plus the charge array.
Writes collective/results/csr3d/<tag>_n<N>_bunches.npz

usage (from the repository root):  python collective/csr3d_dump.py [0.94,1.0,1.06] [1e6]
"""
import os, sys, copy, logging
import numpy as np
from track import b, rs as run_scan, OUT as TRACK_OUT

OUT = os.path.join(TRACK_OUT, "csr3d")
S_CASES = (0.94, 1.0, 1.06)


def segments():
    """List of (name, [elements]) ending at each mid-dipole point and at Z_STOP."""
    from ocelot import SBend, Drift
    a, L = b.THETA, b.RHO * b.THETA
    dlong = (b.Z_DIP[1] - b.Z_DIP[0] - b.LZ) / np.cos(a)
    # (sign of bend, e1 of the first half, e2 of the second half) as in collective.lattice()
    dip = {"B1": (+1, 0.0, a), "B2": (-1, -a, 0.0), "B3": (-1, 0.0, -a), "B4": (+1, a, 0.0)}
    half = lambda n, first: SBend(l=L / 2, angle=dip[n][0] * a / 2,
                                  e1=dip[n][1] if first else 0.0, e2=0.0 if first else dip[n][2])
    return [("B1", [Drift(l=b.Z_DIP[0]), half("B1", True)]),
            ("B2", [half("B1", False), Drift(l=dlong), half("B2", True)]),
            ("B3", [half("B2", False), Drift(l=b.Z_DIP[2] - b.Z_DIP[1] - b.LZ), half("B3", True)]),
            ("B4", [half("B3", False), Drift(l=dlong), half("B4", True)]),
            ("end", [half("B4", False), Drift(l=b.Z_STOP - b.Z_EXIT)])]


def dump(s, n=1_000_000):
    """n: number of macro-particles (1e6 is needed for a converged 3D wake; 1e5 is too noisy)."""
    logging.disable(logging.WARNING)
    from ocelot import MagneticLattice, SecondTM, Navigator, track
    from ocelot.adaptors.astra2ocelot import astraBeam2particleArray
    h = run_scan.H_FULL * s
    tag = f"h{h:07.3f}".replace(".", "p")
    b.NMACRO = n
    os.makedirs(OUT, exist_ok=True)
    ini = os.path.join(OUT, f"{tag}_n{n}.ini")
    np.savetxt(ini, b.make_distribution(h), fmt="%.6E %.6E %.6E %.6E %.6E %.6E %.6E %.6E %d %d")
    tag += f"_n{n}"
    p = astraBeam2particleArray(ini, print_params=False)
    os.remove(ini)
    out = {}
    for name, seq in segments():
        lat = MagneticLattice(seq, method={"global": SecondTM})
        track(lat, p, Navigator(lat), print_progress=False, calc_tws=False)
        out[name] = np.vstack([p.x(), p.y(), p.tau(), p.p(), p.q_array])
        print(f"s={s:.2f} {name}: sigma_tau={p.tau().std()*1e6:.2f} um, sigma_x={p.x().std()*1e3:.3f} mm", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"{tag}_bunches.npz"), s=s, E0=p.E, **out)


if __name__ == "__main__":
    cases = list(map(float, sys.argv[1].split(","))) if len(sys.argv) > 1 and sys.argv[1] else S_CASES
    n = int(float(sys.argv[2])) if len(sys.argv) > 2 else 1_000_000
    for s in cases:
        dump(s, n)
