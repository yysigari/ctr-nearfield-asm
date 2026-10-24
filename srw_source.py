"""Single-electron near-field (edge + synchrotron) radiation from the last chicane dipole, computed with SRW."""
import numpy as np
from srwpy.srwlib import (SRWLMagFldM, SRWLMagFldC, SRWLPartBeam, SRWLWfr, SRWLPrtTrj, srwl, array)
import beamline as b

THZ_EV = 4.135667696e-3            # photon energy of 1 THz [eV]
Z_C = b.Z_DIP[3] + b.LZ / 2        # centre of dipole 4
Z_START = b.Z_DIP[3] - 0.10        # integration starts 10 cm upstream of dipole 4
Z_END = b.Z_EXIT + 0.15
B4 = b.P0 / 299792458.0 / b.RHO    # |B| [T]
LEDGE = 1.1 * b.GAP                # 10-90 % fringe length matching ASTRA's taper
X_CORR = 0.0


def mag_field(sign=1.0):
    d = SRWLMagFldM(_G=sign * B4, _m=1, _n_or_s='n', _Leff=b.LZ, _Ledge=LEDGE)
    return SRWLMagFldC([d], array('d', [0]), array('d', [0]), array('d', [Z_C]))


def part_beam():
    pb = SRWLPartBeam()
    pb.Iavg = 1.0                                   # [A]; rescaled to one electron later
    pb.partStatMom1.x = -b.RHO * (1 - np.cos(b.THETA)) - (b.Z_DIP[3] - Z_START) * np.tan(b.THETA) - X_CORR
    pb.partStatMom1.xp = np.tan(b.THETA)            # electron arrives from the chicane dog-leg
    pb.partStatMom1.z = Z_START
    pb.partStatMom1.gamma = b.GAMMA
    return pb


def trajectory(sign=1.0, n=4000):
    tr = SRWLPrtTrj()
    tr.partInitCond = part_beam().partStatMom1
    tr.allocate(n, True)
    tr.ctStart, tr.ctEnd = 0, Z_END - Z_START
    srwl.CalcPartTraj(tr, mag_field(sign), [1])
    return np.array(tr.arZ), np.array(tr.arX), np.array(tr.arXp)


def field(z_obs, f_thz, half_w, n, sign=1.0, rel_prec=0.005):
    """Complex Ex, Ey on an n x n grid of half-width half_w [m] at z_obs, for frequencies f_thz.
    Returns (Ex, Ey) with shape (nf, n, n) in units sqrt(J / (Hz m^2)) per electron, i.e.
    |Ex|^2 + |Ey|^2 = d^2W / (d nu dA) of a single electron."""
    f_thz = np.atleast_1d(f_thz)
    w = SRWLWfr()
    w.allocate(len(f_thz), n, n)
    w.mesh.zStart = z_obs
    w.mesh.eStart, w.mesh.eFin = f_thz[0] * THZ_EV, f_thz[-1] * THZ_EV
    w.mesh.xStart, w.mesh.xFin = -half_w, half_w
    w.mesh.yStart, w.mesh.yFin = -half_w, half_w
    w.partBeam = part_beam()
    srwl.CalcElecFieldSR(w, 0, mag_field(sign), [2, rel_prec, Z_START, Z_END, 20000, 1, 0])
    ne = len(f_thz)
    ex = np.array(w.arEx, dtype=np.float64).reshape(n, n, ne, 2)   # SRW order: e fastest, then x, then y
    ey = np.array(w.arEy, dtype=np.float64).reshape(n, n, ne, 2)
    Ex = (ex[..., 0] + 1j * ex[..., 1]).transpose(2, 0, 1)          # -> (ne, y, x)
    Ey = (ey[..., 0] + 1j * ey[..., 1]).transpose(2, 0, 1)
    # SRW intensity [ph/s/0.1%bw/mm^2] at Iavg=1 A  ->  per electron energy density per Hz per m^2
    e_ch, hbar = 1.602176634e-19, 1.054571817e-34
    scale = e_ch / 1.0 * hbar * 2 * np.pi / 1e-3 * 1e6 / (2 * np.pi)   # ph/0.1%bw -> J/Hz ; mm^-2 -> m^-2
    return Ex * np.sqrt(scale), Ey * np.sqrt(scale)


def _close_orbit():
    """Tune |B| and entry x so the SRW electron leaves dipole 4 on axis with zero angle (as in ASTRA)."""
    global B4, X_CORR
    from scipy.optimize import brentq
    b0 = B4
    def exit_xp(k):
        global B4
        B4 = b0 * k
        return trajectory()[2][-1]
    B4 = b0 * brentq(exit_xp, 0.98, 1.02)
    X_CORR = trajectory()[1][-1]


_close_orbit()
