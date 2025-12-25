import numpy as np
import mpmath as mp
import libstempo as tempo
from typing import Tuple, Literal
from astropy import units as u
from ..constants import MP_DPS, DAY_TO_SECONDS
from timkf import NP_LONGDOUBLE_TYPE


class PulsarDataLoader:
    def __init__(self, dtype, software: str = "tempo"):
        self.supported_software = ["tempo"]
        self.supported_dtypes = ["mpmath", "numpy"]

        self.software = software
        self._validate_software()
        self.dtype = dtype
        self._validate_dtype()

    def _validate_software(self):
        if self.software not in self.supported_software:
            raise ValueError(
                f"Software {self.software} not supported. Currently only supported: {', '.join(self.supported_software)}"
            )

    def _validate_dtype(self):
        if self.dtype not in self.supported_dtypes:
            raise ValueError(
                f"Data type {self.dtype} not supported. Currently only supported: {', '.join(self.supported_dtypes)}"
            )

    def load(self, parfile: str, timfile: str):
        mp.dps = MP_DPS
        print(f"{mp.dps=}")
        psr = tempo.tempopulsar(parfile=parfile, timfile=timfile)
        # sorted index
        isort = psr.pets().argsort()
        # pulse emission times (pets) is in units of days
        pets = psr.pets()[isort] * mp.mpf(1)
        # time in seconds, shifted so that the first pet is 0
        times = (pets - pets[0]) * DAY_TO_SECONDS

        # phase \propto pulse_number
        phases = psr.pulse_number[isort] * mp.mpf(1)
        # phase relative to the first pulse (i.e. set the phase of the first pulse to 0)
        phases -= phases[0]

        # f_0 is in units of Hz (frequency, not the angular frequency)
        f_0 = psr["F0"].val * mp.mpf(1)
        # f_dot is in units of Hz/s
        f_dot = psr["F1"].val * mp.mpf(1)
        # PEPOCH is in units of days (MJD)
        PEPOCH = psr["PEPOCH"].val * mp.mpf(1)
        # update f_0 deterministically to the first pet
        f_0 = f_0 + f_dot * (pets[0] - PEPOCH) * DAY_TO_SECONDS

        # toaerrs are in units of microseconds -> convert to seconds
        toa_errors = u.us.to(u.s, psr.toaerrs[isort] * mp.mpf(1))
        # phase measurement error (to leading order)
        R_phase = (toa_errors * f_0) ** 2

        # convert to mp.mpf so that precision is not lost
        times = times * mp.mpf(1)
        phases = phases * mp.mpf(1)
        R_phase = R_phase * mp.mpf(1)
        f_0 = f_0 * mp.mpf(1)
        f_dot = f_dot * mp.mpf(1)
        # now multiply by 2 pi where needed
        phases *= 2 * mp.pi
        R_phase *= (2 * mp.pi) ** 2
        omgc0_par = f_0 * 2 * mp.pi
        omgc_dot_par = f_dot * 2 * mp.pi

        return self._package_data(times, phases, R_phase, omgc0_par, omgc_dot_par, psr)

    def _package_data(
        self, times, phases, R_phase, omgc0_par, omgc_dot_par, psr
    ) -> Tuple[
        Tuple[Literal["times"], Literal["phases"], Literal["R_phase"]],
        Tuple[Literal["omgc0_par"], Literal["omgc_dot_par"]],
        Tuple[Literal["tempopulsar"], Literal["tempo2_residuals"]],
    ]:
        """
        TODO: maybe turn this into a dtype coverter
        """
        # tempo2 residuals
        tempo2_residuals = psr.phaseresiduals() * (2 * mp.pi)
        if self.dtype == "numpy":
            # convert to numpy arrays
            times = np.array(times, dtype=NP_LONGDOUBLE_TYPE)
            phases = np.array(phases, dtype=NP_LONGDOUBLE_TYPE)
            R_phase = np.array(R_phase, dtype=NP_LONGDOUBLE_TYPE)
            omgc0_par = NP_LONGDOUBLE_TYPE(omgc0_par)
            omgc_dot_par = NP_LONGDOUBLE_TYPE(omgc_dot_par)
            tempo2_residuals = np.array(tempo2_residuals, dtype=NP_LONGDOUBLE_TYPE)
        return (
            (times, phases, R_phase),
            (omgc0_par, omgc_dot_par),
            (psr, tempo2_residuals),
        )
