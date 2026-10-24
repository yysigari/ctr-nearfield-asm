"""Shared parameters + ASTRA chicane driver for the bunch-compression / near-field radiation study."""
import os, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNDIR = os.path.join(HERE, "astra_runs")
ASTRA = os.path.expanduser("~/bin/astra")

# ---- beam -------------------------------------------------------------------
MC2 = 0.51099895          # MeV
EKIN = 30.0               # MeV
GAMMA = 1 + EKIN / MC2
BETA = np.sqrt(1 - 1 / GAMMA**2)
P0 = np.sqrt((EKIN + MC2)**2 - MC2**2) * 1e6   # eV/c
Q_PC = 20.0               # bunch charge [pC]
NE = Q_PC * 1e-12 / 1.602176634e-19            # real electrons
NMACRO = 100_000
SIG_X = 0.3e-3            # m
EPS_N = 1.0e-6            # m rad (normalised)
SIG_Z0 = 0.3e-3           # m  (1 ps rms)
SIG_DELTA_UNC = 1.0e-4    # uncorrelated relative energy spread

# ---- chicane (rectangular magnets, faces normal to z) ------------------------
THETA = np.deg2rad(7.0)   # bend angle per magnet
LZ = 0.20                 # projected magnet length along z [m]
RHO = LZ / np.sin(THETA)
GAP = 0.01                # ASTRA fringe taper parameter
Z_DIP = [0.20, 1.40, 2.00, 3.20]     # entrance face of each dipole
Z_EXIT = Z_DIP[-1] + LZ              # 3.40 m
Z_STOP = Z_EXIT + 0.05


def make_distribution(h, seed=1):
    """Gaussian 6D bunch with linear chirp delta = -h*z (tail has more energy -> compresses)."""
    rng = np.random.default_rng(seed)
    n = NMACRO
    sxp = EPS_N / (GAMMA * BETA * SIG_X)
    x, y = rng.normal(0, SIG_X, (2, n))
    xp, yp = rng.normal(0, sxp, (2, n))
    z = rng.normal(0, SIG_Z0, n)
    delta = -h * z + rng.normal(0, SIG_DELTA_UNC, n)
    p = P0 * (1 + delta)
    pz = p / np.sqrt(1 + xp**2 + yp**2)
    d = np.zeros((n, 10))
    d[:, 0], d[:, 1], d[:, 2] = x, y, z
    d[:, 3], d[:, 4], d[:, 5] = xp * pz, yp * pz, pz - P0
    d[:, 7] = -Q_PC * 1e-3 / n          # nC per macro
    d[:, 8], d[:, 9] = 1, 5
    d[0] = [0, 0, 0, 0, 0, P0, 0, -Q_PC * 1e-3 / n, 1, 5]   # reference particle, absolute
    return d


def dipole_block():
    lines = ["&DIPOLE", " LDipole=T"]
    sgn = [1, -1, -1, 1]   # +R bends to -x (ASTRA convention)
    for i, (z0, s) in enumerate(zip(Z_DIP, sgn), 1):
        lines += [f" D_Type({i})='hor', D1({i})=(0.5,{z0:.4f}), D2({i})=(-0.5,{z0:.4f}),"
                  f" D3({i})=(0.5,{z0+LZ:.4f}), D4({i})=(-0.5,{z0+LZ:.4f}),",
                  f" D_radius({i})={s*RHO:.6f}, D_Gap(1,{i})={GAP}, D_Gap(2,{i})={GAP}"]
    return "\n".join(lines + ["/"])


def astra_input(tag):
    return f"""&NEWRUN
 Head='chicane {tag}'
 RUN=1, Distribution='{tag}.ini', Xoff=0.0, Yoff=0.0
 TRACK_ALL=T, Auto_Phase=F, H_max=0.002, H_min=0.0
 ZSTART=0.0, ZSTOP={Z_STOP}
 Zemit=170, Zphase=1
 RefS=T, EmitS=T, PhaseS=T, TrackS=F, LandFS=F
/
&CHARGE LSPCH=F /
&APERTURE /
&CAVITY LEFieLD=F /
&SOLENOID LBFieLD=F /
{dipole_block()}
"""


def run_astra(h, tag=None, force=False):
    """Track one chirp setting; returns final ASTRA distribution (ref row first, others relative)."""
    os.makedirs(RUNDIR, exist_ok=True)
    tag = tag or f"h{h:07.3f}".replace(".", "p")
    out = os.path.join(RUNDIR, f"{tag}.{int(round(Z_STOP*100)):04d}.001")
    if force or not os.path.exists(out):
        np.savetxt(os.path.join(RUNDIR, f"{tag}.ini"), make_distribution(h),
                   fmt="%.6E %.6E %.6E %.6E %.6E %.6E %.6E %.6E %d %d")
        with open(os.path.join(RUNDIR, f"{tag}.in"), "w") as f:
            f.write(astra_input(tag))
        subprocess.run([ASTRA, f"{tag}.in"], cwd=RUNDIR, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.loadtxt(out)


def bunch_coordinates(d):
    """x, y [m], arrival delay t [s] (later = positive) and delta for alive particles."""
    ref = d[0]
    good = d[:, 9] >= 0
    good[0] = True
    d = d[good].copy()
    z = d[:, 2].copy(); z[0] = 0.0
    pz = d[:, 5].copy(); pz[0] = 0.0
    p = np.sqrt(d[:, 3]**2 + d[:, 4]**2 + (ref[5] + pz)**2)
    x, y = d[:, 0] - ref[0], d[:, 1] - ref[1]
    t = -z / (BETA * 299792458.0)
    return x, y, t, p / P0 - 1
