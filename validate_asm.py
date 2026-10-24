"""Validate ASM against SRW: SRW field at z1 propagated by ASM vs SRW computed directly at z2."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import beamline as b, srw_source as s, asm

N, HW = 384, 0.064
F = np.array([0.3, 0.6, 1.2, 2.4])
z1, z2 = b.Z_EXIT + 0.5, b.Z_EXIT + 2.0
x = np.linspace(-HW, HW, N); dx = x[1] - x[0]
E1 = s.field(z1, F, HW, N); E2 = s.field(z2, F, HW, N)
P = [asm.propagate(E1[c], dx, F * 1e12, z2 - z1) for c in range(2)]
I_srw = abs(E2[0])**2 + abs(E2[1])**2
I_asm = abs(P[0])**2 + abs(P[1])**2
fig, ax = plt.subplots(3, len(F), figsize=(4 * len(F), 10.5))
ext = [x[0] * 1e3, x[-1] * 1e3] * 2
for j, f in enumerate(F):
    vmax = I_srw[j].max()
    ax[0, j].imshow(I_srw[j], extent=ext, origin="lower", vmax=vmax, cmap="inferno"); ax[0, j].set_title(f"SRW direct, {f} THz, z={z2-b.Z_EXIT:.1f} m")
    ax[1, j].imshow(I_asm[j], extent=ext, origin="lower", vmax=vmax, cmap="inferno"); ax[1, j].set_title("SRW(z=0.5 m) + ASM")
    mid = N // 2
    ax[2, j].plot(x * 1e3, I_srw[j][mid], label="SRW direct"); ax[2, j].plot(x * 1e3, I_asm[j][mid], "--", label="ASM")
    ax[2, j].plot(x * 1e3, I_srw[j][:, mid], label="SRW direct (y cut)"); ax[2, j].plot(x * 1e3, I_asm[j][:, mid], "--", label="ASM (y cut)")
    ax[2, j].set_xlabel("x or y [mm]")
    inner = np.abs(x) < 0.04
    m = np.ix_(inner, inner)
    err = np.sqrt(((I_asm[j][m] - I_srw[j][m])**2).sum() / (I_srw[j][m]**2).sum())
    ax[2, j].set_title(f"rel. L2 error (|x|,|y|<40 mm): {err:.3f}")
    print(f, "THz  err", err)
ax[2, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig("media/validation_asm_vs_srw.png", dpi=90)
