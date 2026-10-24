"""Angular Spectrum Method: exact scalar free-space propagation of each transverse field component."""
import numpy as np

C = 299792458.0


def propagate(E, dx, f_hz, dz, pad=2, window=True, band_limit=True):
    """Propagate complex field(s) E[..., ny, nx] (one per frequency in f_hz) by dz [m].

    E(kx,ky; z+dz) = E(kx,ky; z) * exp(i dz sqrt(k^2 - kx^2 - ky^2)), evanescent waves dropped.
    Zero-padding by `pad` suppresses FFT wrap-around; a soft super-Gaussian edge taper
    removes diffraction from the hard truncation of the source window.
    band_limit: Matsushima & Shimobaba (2009) band-limited ASM; drops spatial frequencies whose
    transfer-function phase is under-sampled on the padded grid (they would leave the window anyway)."""
    E = np.asarray(E, complex)
    *lead, ny, nx = E.shape
    f_hz = np.broadcast_to(np.atleast_1d(f_hz), (E.shape[0],) if lead else (1,))
    if window:
        u = np.linspace(-1, 1, nx)
        w1 = np.exp(-(np.abs(u) / 0.92) ** 24)
        E = E * np.outer(w1, w1)
    Ny, Nx = pad * ny, pad * nx
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dx)
    KX2, KY2 = np.meshgrid(kx**2, ky**2)
    y0, x0 = (Ny - ny) // 2, (Nx - nx) // 2
    out = np.empty_like(E)
    Ef = E.reshape(-1, ny, nx)
    for i, f in enumerate(f_hz if lead else [f_hz[0]]):
        k = 2 * np.pi * f / C
        kz2 = k**2 - KX2 - KY2
        H = np.where(kz2 > 0, np.exp(1j * dz * np.sqrt(np.abs(kz2))), 0)
        if band_limit:
            u_lim = 1 / (C / f * np.sqrt((2 * abs(dz) / (Nx * dx))**2 + 1))      # cycles / m
            H = H * ((KX2 <= (2 * np.pi * u_lim)**2) & (KY2 <= (2 * np.pi * u_lim)**2))
        buf = np.zeros((Ny, Nx), complex)
        buf[y0:y0 + ny, x0:x0 + nx] = Ef[i]
        buf = np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(buf)) * H)
        out.reshape(-1, ny, nx)[i] = np.fft.fftshift(buf)[y0:y0 + ny, x0:x0 + nx]
    return out
