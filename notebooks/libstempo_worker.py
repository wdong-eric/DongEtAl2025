import os
import gc
import tempfile
import numpy as np
from pathlib import Path

K_DM = 4.149  # ms GHz^2 pc^-1 cm^3
MHz_to_GHz = 1e-3
MICROSEC_TO_MSEC = 1e-3
SEC_TO_MSEC = 1e3


def delDM_bound_from_timing_noise(tempo_psr):
    # toaerrs = tempo_psr.toaerrs * MICROSEC_TO_MSEC  # convert to ms
    # print(f"min, max TOA errors (ms): {toaerrs.min()}, {toaerrs.max()}")
    residuals = tempo_psr.residuals() * SEC_TO_MSEC  # in ms
    # print(
    #     f"min, max Residuals (ms): {np.min(np.abs(residuals))}, {np.max(np.abs(residuals))}"
    # )
    freq_bands = tempo_psr.freqs * MHz_to_GHz  # convert to GHz
    # print(f"Median Frequencies (GHz): {np.median(freq_bands)}")
    DM_variations = residuals / (K_DM * freq_bands**-2)
    # print(f"max required DM variations (pc cm^-3): {np.abs(DM_variations).max()}")
    return DM_variations


def worker(par: str, tim: str):
    import libstempo

    print(f"Working on {Path(par).name}, {Path(tim).name}")
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            psr = libstempo.tempopulsar(parfile=par, timfile=tim)
            dDM = delDM_bound_from_timing_noise(psr)
            return np.abs(dDM).max(), (psr["START"].val, psr["FINISH"].val)
        finally:
            try:
                del psr
            except NameError:
                pass
            os.chdir(cwd)
            gc.collect()
