"""Coherent transition radiation (CTR) from a finite, ideally conducting foil, built from ASTRA particles.

Source plane (foil):  each electron at (x_j, y_j) arriving at t_j contributes the Ginzburg-Frank field
    E_r(r, w) = e w / (2 pi eps0 beta^2 c^2 gamma) * K1(k r / (beta gamma)),   E(w) = int E(t) e^{iwt} dt
so the bunch field is the convolution of that kernel with the complex transverse density
    S_w(x, y) = sum_j delta(x - x_j) delta(y - y_j) e^{i w t_j}
truncated by the foil aperture. The radiated field is then propagated with the ASM (asm.py).
Spectral fluence: d^2W / (d nu dA) = 2 eps0 c |E(w)|^2.
"""
import numpy as np
from scipy.special import k1
import beamline as b

C, EPS0, QE = 299792458.0, 8.8541878128e-12, 1.602176634e-19


class Grid:
    def __init__(self, n=256, half_width=0.064):
        self.n, self.hw = n, half_width
        self.x = (np.arange(n) - n // 2) * (2 * half_width / n)
        self.dx = self.x[1] - self.x[0]
        self.X, self.Y = np.meshgrid(self.x, self.x)


def kernel_q(f_hz, n_big, dx):
    """Single-electron foil field (Ex, Ey) built analytically in k-space (no aliasing of the 1/r core):
    FT{E_r(r) r_hat} = -2 pi i A q_vec / (alpha (q^2 + alpha^2)),  alpha = k / (beta gamma).
    Returned in real space on an n_big x n_big grid centred at index n_big//2."""
    w = 2 * np.pi * f_hz
    alpha = w / C / (b.BETA * b.GAMMA)
    A = QE * w / (2 * np.pi * EPS0 * b.BETA**2 * C**2 * b.GAMMA)
    q = 2 * np.pi * np.fft.fftfreq(n_big, dx)
    QX, QY = np.meshgrid(q, q)
    common = -2j * np.pi * A / (alpha * (QX**2 + QY**2 + alpha**2)) / dx**2
    ex = np.fft.fftshift(np.fft.ifft2(common * QX))
    ey = np.fft.fftshift(np.fft.ifft2(common * QY))
    return ex, ey


def deposit(g, x, y, phase_w):
    """Cloud-in-cell deposit of complex weights phase_w (normalised to sum 1) onto the grid."""
    fx = (x - g.x[0]) / g.dx; fy = (y - g.x[0]) / g.dx
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - ix, fy - iy
    out = np.zeros(g.n * g.n, complex)
    for ddx, wx in ((0, 1 - tx), (1, tx)):
        for ddy, wy in ((0, 1 - ty), (1, ty)):
            jx, jy = ix + ddx, iy + ddy
            ok = (jx >= 0) & (jx < g.n) & (jy >= 0) & (jy < g.n)
            idx = jy[ok] * g.n + jx[ok]
            ww = (wx * wy * phase_w)[ok]
            out += np.bincount(idx, ww.real, g.n * g.n) + 1j * np.bincount(idx, ww.imag, g.n * g.n)
    return out.reshape(g.n, g.n)


def bunch_field(g, f_hz, x, y, t, n_e=b.NE, foil_radius=0.020):
    """Total coherent foil field (Ex, Ey)[nf, n, n] of a bunch of n_e electrons represented by macro-particles."""
    f_hz = np.atleast_1d(f_hz)
    n = g.n
    foil = (g.X**2 + g.Y**2) <= foil_radius**2
    Ex = np.zeros((len(f_hz), n, n), complex); Ey = np.zeros_like(Ex)
    for i, f in enumerate(f_hz):
        S = deposit(g, x, y, np.exp(1j * 2 * np.pi * f * t) / len(t))
        kx, ky = kernel_q(f, 2 * n, g.dx)
        Sp = np.zeros((2 * n, 2 * n), complex); Sp[:n, :n] = S
        Sf = np.fft.fft2(Sp)
        # linear convolution: kernel centred at index n -> output offset n//2 aligns with grid centre
        for K, out in ((kx, Ex), (ky, Ey)):
            conv = np.fft.ifft2(Sf * np.fft.fft2(np.fft.ifftshift(K)))
            conv = np.roll(conv, (n // 2, n // 2), axis=(0, 1))[n // 2: n // 2 + n, n // 2: n // 2 + n]
            out[i] = n_e * conv * foil
    return Ex, Ey


def fluence(Ex, Ey):
    """d^2W/(d nu dA) [J / (Hz m^2)]."""
    return 2 * EPS0 * C * (np.abs(Ex)**2 + np.abs(Ey)**2)


def form_factor(t, f_hz, bins=4000):
    """|F(f)|^2 of the longitudinal profile from a histogram (suppresses macro-particle shot noise)."""
    h, e = np.histogram(t, bins=bins)
    tc = 0.5 * (e[1:] + e[:-1])
    F = (h[None, :] * np.exp(1j * 2 * np.pi * np.atleast_1d(f_hz)[:, None] * tc[None, :])).sum(1) / h.sum()
    return np.abs(F)**2
