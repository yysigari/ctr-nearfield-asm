# WarpX (exploratory)

`drift_test.py` tracks the 30 MeV, 20 pC bunch through a straight drift in 3D WarpX (moving window at c), as a first
step towards a full electromagnetic CSR check of one chicane dipole. Requires WarpX ≥ 26.10 (`conda install -c
conda-forge warpx`).

```bash
OMP_NUM_THREADS=4 python warpx/drift_test.py ckc 30      # solver: psatd | ckc | yee, number of steps
```

Notes from running it on a 4-core laptop CPU (4.6 M cells, 10⁵ particles):
* With a moving window, use small boxes (`warpx_max_grid_size=64`); with one large box the window shift runs on
  one thread and takes ~90 % of the time (CKC: 8.4 → 1.7 s/step). PSATD on one box: 26 s/step.
* `initialize_self_field=True` with open (PML) field boundaries needs the FFT Poisson solver
  (`pywarpx.warpx.poisson_solver = "fft"`).
* The projected cost of the single-dipole 3D test (~20 M cells, ~8·10⁴ steps) is about a week on such a machine,
  so the 3D CSR check in `collective/` uses PyCSR3D instead. The full chicane at full compression is out of reach
  without a GPU or a cluster (the beam moves 147 mm in x and slips 17.5 mm behind a window moving at c).
