"""WarpX step 1: 30 MeV, 20 pC Gaussian bunch in a straight drift (3D, moving window at c).

Checks (a) that the solver does not radiate numerically (no Cherenkov / no field ahead of the bunch)
and (b) the throughput on this machine, to project the cost of the single-dipole CSR test.

usage: OMP_NUM_THREADS=4 python drift_test.py <solver: psatd|ckc|yee> <nsteps>
"""
import sys, time, os
import numpy as np
from pywarpx import picmi
import pywarpx
from scipy.constants import c, e, m_e

SOLVER = sys.argv[1] if len(sys.argv) > 1 else "psatd"
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300

EKIN = 30e6
GAMMA = 1 + EKIN * e / (m_e * c**2)
BETA = np.sqrt(1 - 1 / GAMMA**2)
Q = 20e-12
SIG_X, SIG_Z = 0.3e-3, 100e-6
EPS_N = 1e-6
NMACRO = 100_000

# grid: +-8 sigma_x transversely, +-8 sigma_z longitudinally
NX, NZ = 120, 320
XMAX, ZHALF = 2.4e-3, 0.8e-3
grid = picmi.Cartesian3DGrid(
    number_of_cells=[NX, NX, NZ],
    lower_bound=[-XMAX, -XMAX, -ZHALF], upper_bound=[XMAX, XMAX, ZHALF],
    lower_boundary_conditions=["open"] * 3, upper_boundary_conditions=["open"] * 3,
    lower_boundary_conditions_particles=["absorbing"] * 3,
    upper_boundary_conditions_particles=["absorbing"] * 3,
    moving_window_velocity=[0, 0, c], warpx_max_grid_size=64, warpx_blocking_factor=8)

if SOLVER == "psatd":
    solver = picmi.ElectromagneticSolver(grid=grid, method="PSATD", cfl=0.99, warpx_psatd_update_with_rho=True)
else:
    solver = picmi.ElectromagneticSolver(grid=grid, method=SOLVER.upper(), cfl=0.99)

ux = EPS_N / SIG_X                    # rms u_x = gamma*beta*sigma_x' for a waist with emittance EPS_N
dist = picmi.GaussianBunchDistribution(
    n_physical_particles=Q / e, rms_bunch_size=[SIG_X, SIG_X, SIG_Z],
    rms_velocity=[ux * c, ux * c, 1e-4 * GAMMA * c],
    centroid_position=[0, 0, 0], centroid_velocity=[0, 0, GAMMA * BETA * c])
beam = picmi.Species(particle_type="electron", name="beam", initial_distribution=dist)

sim = picmi.Simulation(solver=solver, max_steps=NSTEPS, verbose=1, particle_shape="cubic",
                       warpx_use_filter=1, warpx_do_dynamic_scheduling=0)
sim.add_species(beam, layout=picmi.PseudoRandomLayout(n_macroparticles=NMACRO, grid=grid),
                initialize_self_field=True)

pywarpx.warpx.poisson_solver = "fft"      # open-boundary (IGF) solver for the initial self-field

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", f"drift_{SOLVER}")
sim.add_diagnostic(picmi.FieldDiagnostic(grid=grid, period=NSTEPS, data_list=["Ex", "Ey", "Ez", "Bx", "By"],
                                         write_dir=out, name="fld", warpx_format="openpmd"))
sim.add_diagnostic(picmi.ParticleDiagnostic(period=NSTEPS, species=[beam], write_dir=out, name="ptl",
                                            warpx_format="openpmd"))

t0 = time.time()
sim.step(1)
t1 = time.time()
sim.step(NSTEPS - 1)
t2 = time.time()
ncell = NX * NX * NZ
per = (t2 - t1) / (NSTEPS - 1)
print(f"\nSOLVER {SOLVER}: init+1 step {t1-t0:.1f} s, {per*1e3:.1f} ms/step, "
      f"{ncell*(NSTEPS-1)/(t2-t1)/1e6:.1f} Mcell-steps/s, {NMACRO*(NSTEPS-1)/(t2-t1)/1e6:.2f} Mpart-steps/s")
