"""Validate ASM propagation of the CTR field against (a) semi-analytic Fresnel-Hankel integral and (b) SRW's own propagator."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import k1, j1
from scipy.integrate import quad_vec
import beamline as b, ctr, asm
from srwpy.srwlib import SRWLWfr, SRWLOptD, SRWLOptC, srwl, array

A = 0.020
g = ctr.Grid(256, 0.064)
F = [0.3, 1.0, 2.5]           # THz
Z = [0.1, 0.5, 2.0]           # m from foil
fig, ax = plt.subplots(len(F), len(Z), figsize=(13, 10), sharex=True)


def hankel(f, z, rho):
    """Paraxial Fresnel propagation of the radial K1 field of a single electron through a circular foil."""
    k = 2 * np.pi * f / ctr.C
    amp = ctr.QE * 2 * np.pi * f / (2 * np.pi * ctr.EPS0 * b.BETA**2 * ctr.C**2 * b.GAMMA)
    g_ = lambda r: amp * k1(k * r / (b.BETA * b.GAMMA)) * j1(k * rho * r / z) * np.exp(1j * k * r**2 / (2 * z)) * r
    val, _ = quad_vec(g_, 1e-7, A, limit=2000, epsabs=0)
    return np.abs(k / z * val)**2


def srw_prop(Ex, Ey, f, z):
    w = SRWLWfr(); w.allocate(1, g.n, g.n)
    w.mesh.eStart = w.mesh.eFin = f * 4.135667696e-3
    w.mesh.xStart, w.mesh.xFin = g.x[0], g.x[-1]; w.mesh.yStart, w.mesh.yFin = g.x[0], g.x[-1]
    w.mesh.zStart = 0; w.Rx = w.Ry = 1e23
    w.arEx = array('f', np.stack([Ex.real, Ex.imag], -1).astype(np.float32).ravel())
    w.arEy = array('f', np.stack([Ey.real, Ey.imag], -1).astype(np.float32).ravel())
    # drift; prop. params: [auto-resize, ..., range x, res x, range y, res y]
    srwl.PropagElecField(w, SRWLOptC([SRWLOptD(z)], [[0, 0, 1.0, 0, 0, 3.0, 1.0, 3.0, 1.0, 0, 0, 0]]))
    nx, ny = w.mesh.nx, w.mesh.ny
    ex = np.array(w.arEx).reshape(ny, nx, 2); ey = np.array(w.arEy).reshape(ny, nx, 2)
    I = 2 * ctr.EPS0 * ctr.C * (ex[..., 0]**2 + ex[..., 1]**2 + ey[..., 0]**2 + ey[..., 1]**2)
    xs = np.linspace(w.mesh.xStart, w.mesh.xFin, nx)
    return xs, I[ny // 2]


for i, f in enumerate(F):
    Ex, Ey = ctr.bunch_field(g, f * 1e12, np.array([0.0]), np.array([0.0]), np.array([0.0]), n_e=1, foil_radius=A)
    for j, z in enumerate(Z):
        Px = asm.propagate(Ex[0], g.dx, f * 1e12, z, pad=3, window=False)
        Py = asm.propagate(Ey[0], g.dx, f * 1e12, z, pad=3, window=False)
        I_asm = ctr.fluence(Px, Py)[g.n // 2]
        rho = np.linspace(1e-4, g.hw, 200)
        I_h = 2 * ctr.EPS0 * ctr.C * hankel(f * 1e12, z, rho)
        xs, I_srw = srw_prop(Ex[0], Ey[0], f, z)
        a = ax[i, j]
        a.plot(g.x * 1e3, I_asm, label="ASM (this work)")
        a.plot(rho * 1e3, I_h, "k--", lw=1, label="Fresnel-Hankel (semi-analytic)")
        a.plot(xs * 1e3, I_srw, ":", lw=2, label="SRW drift propagator")
        sel = (g.x > 0.5e-3) & (g.x < 0.06)
        err = np.sqrt(np.mean((I_asm[sel] - np.interp(g.x[sel], rho, I_h))**2)) / I_h.max()
        a.set_title(f"{f} THz, z = {z} m  (z/γ²λ = {z / (b.GAMMA**2 * ctr.C / (f*1e12)):.2f})\nrms dev ASM-Hankel = {err:.3f} of peak", fontsize=9)
        a.set_xlim(-64, 64)
        print(f, z, "err", err)
ax[0, 0].legend(fontsize=7)
for a in ax[-1]: a.set_xlabel("x [mm]")
for a in ax[:, 0]: a.set_ylabel("d²W/dνdA [J/Hz/m²]")
fig.suptitle(f"Single-electron CTR, foil radius {A*1e3:.0f} mm, γ={b.GAMMA:.1f}: ASM vs semi-analytic vs SRW")
fig.tight_layout(); fig.savefig("media/validation_ctr_asm.png", dpi=90)
