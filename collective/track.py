"""Track the chirped bunches through the chicane with Ocelot, with and without collective effects.

Beam, chicane and initial distribution come from ../beamline.py and the chirp scan from ../run_scan.py, so the
Ocelot model is the same as the ASTRA one.

Modes:
  none : no collective effects (lattice check against ASTRA)
  sc   : 3D space charge
  csr  : 1D projected CSR (steady state + transients)
  both : sc + csr
Output: collective/results/<tag>_<mode>.ast (ASTRA format) at z = Z_STOP.

Usage:  python collective/track.py [modes] [hR56 values]
        python collective/track.py none,csr,sc,both            # full 32-point scan
        python collective/track.py both 0.94,1.0,1.06          # selected settings
"""
import os, sys, logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import beamline as b, run_scan as rs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODES = ("none", "sc", "csr", "both")


def tag(h):
    return f"h{h:07.3f}".replace(".", "p")


def lattice():
    from ocelot import SBend, Drift, Marker, MagneticLattice, SecondTM
    a, L = b.THETA, b.RHO * b.THETA                      # bend angle, arc length
    dlong = (b.Z_DIP[1] - b.Z_DIP[0] - b.LZ) / np.cos(a)  # path length between B1-B2 and B3-B4
    m0, m1 = Marker(eid="start"), Marker(eid="stop")
    # rectangular magnets with faces normal to z: edge angle 0 on the straight side, +-theta on the tilted side
    seq = [m0, Drift(l=b.Z_DIP[0]),
           SBend(l=L, angle=a, e1=0.0, e2=a, eid="B1"), Drift(l=dlong),
           SBend(l=L, angle=-a, e1=-a, e2=0.0, eid="B2"), Drift(l=b.Z_DIP[2] - b.Z_DIP[1] - b.LZ),
           SBend(l=L, angle=-a, e1=0.0, e2=-a, eid="B3"), Drift(l=dlong),
           SBend(l=L, angle=a, e1=a, e2=0.0, eid="B4"), Drift(l=b.Z_STOP - b.Z_EXIT), m1]
    return MagneticLattice(seq, method={"global": SecondTM}), m0, m1


def initial_file(h):
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, f"{tag(h)}.ini")
    if not os.path.exists(f):
        np.savetxt(f, b.make_distribution(h), fmt="%.6E %.6E %.6E %.6E %.6E %.6E %.6E %.6E %d %d")
    return f


def output_file(h, mode, out=None):
    return os.path.join(out or OUT, f"{tag(h)}_{mode}.ast")


def track(h, mode, nmesh=63, step=0.005, out=None, force=False):
    out_f = output_file(h, mode, out)
    if os.path.exists(out_f) and not force:
        return out_f
    os.makedirs(os.path.dirname(out_f), exist_ok=True)
    logging.disable(logging.WARNING)
    from ocelot import Navigator, CSR, SpaceCharge, track as otrack
    from ocelot.adaptors.astra2ocelot import astraBeam2particleArray, particleArray2astraBeam
    p = astraBeam2particleArray(initial_file(h), print_params=False)
    lat, m0, m1 = lattice()
    navi = Navigator(lat, unit_step=step)
    if mode in ("sc", "both"):
        navi.add_physics_proc(SpaceCharge(step=1, nmesh_xyz=[nmesh] * 3), m0, m1)
    if mode in ("csr", "both"):
        csr = CSR(n_bin=300, m_bin=5, sigma_min=0.2e-6, traj_step=0.0002, apply_step=0.0005)
        navi.add_physics_proc(csr, lat.sequence[2], m1)   # from B1 entrance; transients persist after B4
    otrack(lat, p, navi, print_progress=False, calc_tws=False)
    particleArray2astraBeam(p, out_f)
    return out_f


def _job(args):
    return track(*args)


if __name__ == "__main__":
    modes = sys.argv[1].split(",") if len(sys.argv) > 1 else list(MODES)
    hs = rs.H_LIST if len(sys.argv) < 3 else rs.H_FULL * np.array([float(s) for s in sys.argv[2].split(",")])
    jobs = [(h, m) for m in modes for h in hs]
    with ProcessPoolExecutor(min(8, os.cpu_count() or 1)) as ex:
        for (h, m), f in zip(jobs, ex.map(_job, jobs)):
            print(f"done hR56={h / rs.H_FULL:.3f} {m}", flush=True)
