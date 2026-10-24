"""Run the ASTRA chirp scan (under -> full -> over-compression) in parallel."""
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import beamline as b

R56 = 0.03536                      # measured from h=0 run (dz/ddelta)
H_FULL = 1 / R56
S = np.concatenate([np.linspace(0, 0.84, 9), np.linspace(0.88, 1.12, 17), np.linspace(1.16, 1.40, 6)])
H_LIST = H_FULL * S

if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex:
        for h, _ in zip(H_LIST, ex.map(b.run_astra, H_LIST)):
            print(f"done h={h:.3f}", flush=True)
