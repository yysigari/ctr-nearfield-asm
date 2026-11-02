"""3D vs 1D steady-state CSR wake on the real bunches inside the chicane dipoles (PyCSR3D, Cai & Ding PRAB 23 014402).

Needs PyCSR3D (github.com/ChristopherMayes/PyCSR3D; numpy, scipy, numba, mpmath). From the repository root:
    python collective/csr3d_check.py validate     # thin Gaussian: 3D -> 1D limit
    python collective/csr3d_check.py              # real bunches -> media/collective/csr3d_wakes.pdf, csr3d_numbers.txt
Input: collective/results/csr3d/<tag>_n1000000_bunches.npz from csr3d_dump.py (Ocelot, no collective effects, mid-dipole points).
Ocelot tau is positive at the tail, so z = -tau (z > 0 = head), the convention of PyCSR3D and of the 1D wake.
Both models are steady state, so the ratio 3D/1D measures the transverse-size (Derbenev) effect only, not transients.
"""
import os, sys, glob, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import e, c, physical_constants
from csr3d.wake import green_mesh
from csr3d.convolution import fftconvolve3
from csr3d.csr1d import csr1d_steady_state_kick_calc
from csr3d.dist import lambda_gauss3, lambda_gauss3_prime, gauss

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "csr3d")
FIG = os.path.join(os.path.dirname(HERE), "media", "collective")
R_E = physical_constants["classical electron radius"][0]
MEC2 = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
THETA, LZ = np.deg2rad(7.0), 0.20
RHO = LZ / np.sin(THETA)
BEND_SIGN = {"B1": +1, "B2": -1, "B3": -1, "B4": +1}


def wake3d(density, dprime, deltas, rho, gamma):
    """Steady-state 3D longitudinal wake [1/m^2] on the grid, for a density normalised to 1."""
    G = green_mesh(density.shape, deltas, rho=rho, gamma=gamma, component="s")
    W = fftconvolve3(dprime, G)[0]
    return W * (1 - 1 / gamma**2) / abs(rho) * np.prod(deltas)


def validate():
    """Analytic Gaussians. (1) Thin beam at gamma = 500: 3D must reduce to the 1D steady-state wake.
    (2) Same sigma_z at gamma = 60 (this chicane): finite-gamma terms make 3D differ from 1D even for a thin beam."""
    cases = [(1e-6, 10e-6, 1.0, 500.0, (16, 16, 512)), (10e-6, 10e-6, 1.0, 500.0, (32, 32, 512)),
             (30e-6, 22.6e-6, RHO, 1 + 30e6 / MEC2, (32, 32, 512)), (353e-6, 22.6e-6, RHO, 1 + 30e6 / MEC2, (64, 32, 512))]
    for sx, sz, rho, gamma, n in cases:
        vecs = [np.linspace(-5 * s, 5 * s, k) for s, k in zip((sx, sx, sz), n)]
        d = [v[1] - v[0] for v in vecs]
        X, Y, Z = np.meshgrid(*vecs, indexing="ij")
        dens = lambda_gauss3(X, Y, Z, sigma_x=sx, sigma_y=sx, sigma_z=sz)
        norm = dens.sum() * np.prod(d)
        W = wake3d(dens / norm, lambda_gauss3_prime(X, Y, Z, sigma_x=sx, sigma_y=sx, sigma_z=sz) / norm, d, rho, gamma)
        z1 = np.linspace(-5 * sz, 5 * sz, 4000)
        ss = csr1d_steady_state_kick_calc(z1, gauss(z1, sigma=sz), nz=4000, rho=rho, normalized_units=True)
        w1 = np.interp(vecs[2], ss["zvec"], ss["wake"])
        lz = dens.sum((0, 1)); w3 = (W * dens).sum((0, 1)) / lz          # slice-averaged 3D wake
        print(f"validate gamma={gamma:5.0f} sigma_x={sx*1e6:4.0f} um sigma_z={sz*1e6:4.1f} um "
              f"kappa={sx/sz*np.sqrt(sx/rho):.1e} rho/gamma^3={rho/gamma**3*1e6:.2f} um: "
              f"mean 3D/1D {(w3*lz).sum()/(w1*lz).sum():.3f}, min 3D/1D {w3.min()/w1.min():.3f}, "
              f"argmin z/sigma_z 1D {vecs[2][w1.argmin()]/sz:+.2f} 3D {vecs[2][w3.argmin()]/sz:+.2f}")


def bunch_wakes(P, gamma, rho, n=(64, 32, 512), smooth=1.0):
    """Particle-by-particle energy-change rate [eV/m]: 3D (PyCSR3D) and 1D steady state."""
    x, y, tau, _, q = P
    z = -tau
    Q = q.sum()
    lo = [np.percentile(v, 0.05) for v in (x, y, z)]
    hi = [np.percentile(v, 99.95) for v in (x, y, z)]
    pad = [0.15 * (h - l) for l, h in zip(lo, hi)]
    vecs = [np.linspace(l - p, h + p, k) for l, h, p, k in zip(lo, hi, pad, n)]
    d = [v[1] - v[0] for v in vecs]
    edges = [np.concatenate([v - dd / 2, [v[-1] + dd / 2]]) for v, dd in zip(vecs, d)]
    H, _ = np.histogramdd((x, y, z), bins=edges, weights=q)
    H = gaussian_filter(H, smooth)
    dens = H / (Q * np.prod(d))
    dprime = np.gradient(dens, d[2], axis=2)
    W = wake3d(dens, dprime, d, rho, gamma)
    k3 = RegularGridInterpolator(vecs, W, bounds_error=False, fill_value=0.0)((x, y, z)) * (Q / e) * R_E * MEC2
    k1 = csr1d_steady_state_kick_calc(z, q, nz=300, rho=abs(rho))["denergy_ds"]
    return z, k3, k1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate(); sys.exit()
    files = sorted(glob.glob(os.path.join(OUT, "*_n1000000_bunches.npz")))   # 1e6 particles: converged mean wake
    res = {}
    lines = ["3D/1D steady-state CSR at mid-dipole (Ocelot bunches, no collective effects)",
             " hR56 dip  sig_z[um] sig_x[um] kappa   <dE/ds> 1D  3D [keV/m]  ratio   rms(dE/ds) 1D  3D [keV/m]  ratio"]
    for f in files:
        D = np.load(f)
        s, gamma = float(D["s"]), float(D["E0"]) * 1e9 / MEC2
        for dip in ("B1", "B2", "B3", "B4"):
            P = D[dip]
            t0 = time.time()
            z, k3, k1 = bunch_wakes(P, gamma, BEND_SIGN[dip] * RHO)
            q = P[4]
            m3, m1 = np.average(k3, weights=q), np.average(k1, weights=q)
            r3 = np.sqrt(np.average((k3 - m3)**2, weights=q)); r1 = np.sqrt(np.average((k1 - m1)**2, weights=q))
            sz, sx = z.std(), P[0].std()
            kappa = sx / sz * np.sqrt(sx / RHO)
            res[(s, dip)] = (z, k3, k1, q)
            lines.append(f" {s:4.2f}  {dip}  {sz*1e6:8.2f} {sx*1e6:8.1f}  {kappa:6.2f}   {m1/1e3:9.2f} {m3/1e3:9.2f}  {m3/m1:6.3f}"
                         f"   {r1/1e3:11.2f} {r3/1e3:9.2f}  {r3/r1:6.3f}")
            print(lines[-1], f"({time.time()-t0:.0f} s)", flush=True)
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, "csr3d_numbers.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # wake along the bunch (binned) at B3 and B4 for each chirp
    ss = sorted({k[0] for k in res})
    fig, ax = plt.subplots(2, len(ss), figsize=(7.2, 4.4), squeeze=False)
    for j, s in enumerate(ss):
        for i, dip in enumerate(("B3", "B4")):
            z, k3, k1, q = res[(s, dip)]
            o = np.argsort(z); zz = z[o] * 1e6
            nb = 120
            bins = np.linspace(np.percentile(zz, 0.5), np.percentile(zz, 99.5), nb + 1)
            idx = np.digitize(zz, bins) - 1
            keep = (idx >= 0) & (idx < nb)
            idx = idx[keep]
            cnt = np.bincount(idx, minlength=nb)
            zc = 0.5 * (bins[1:] + bins[:-1])
            for kk, lab, sty in ((k1[o], "1D", "k--"), (k3[o], "3D", "C3-")):
                mean = np.bincount(idx, kk[keep], minlength=nb) / np.maximum(cnt, 1)
                ax[i, j].plot(zc, np.where(cnt > 20, mean / 1e3, np.nan), sty, lw=1, label=lab)
            a2 = ax[i, j].twinx(); a2.fill_between(zc, cnt, color="0.85", zorder=0); a2.set_yticks([])
            ax[i, j].set_zorder(a2.get_zorder() + 1); ax[i, j].patch.set_visible(False)
            ax[i, j].set_title(f"$hR_{{56}}$={s:.2f}, mid-{dip}", fontsize=7)
            ax[i, j].tick_params(labelsize=6)
            if j == 0: ax[i, j].set_ylabel("d$E$/d$s$ [keV/m]", fontsize=7)
            if i == 1: ax[i, j].set_xlabel("z [µm] (head > 0)", fontsize=7)
    ax[0, 0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "csr3d_wakes.pdf"))
    print("\n".join(lines))
