import numpy as np
from ..constants import MILLISECOND, B_CRIT


def period_from_freq(f):
    """
    Calculate the period of a pulsar from its frequency.

    Parameters:
        f: float [Hz]
            Frequency of the pulsar.

    Returns:
        P: float [s]
            Period of the pulsar.
    """
    return 1 / f


def P_dot_from_freq_freq_dot(f, f_dot):
    """
    Calculate the period derivative of a pulsar from its frequency and frequency derivative.

    Parameters:
        f: float [Hz]
            Frequency of the pulsar.
        f_dot: float [Hz/s]
            Frequency derivative of the pulsar.

    Returns:
        P_dot: float [s/s]
            Period derivative of the pulsar.
    """
    return np.abs(f_dot) / (f**2)


def B_surf(omgc, omgc_dot):
    """
    Calculate the surface magnetic field strength of a pulsar
    using the formula from the book Essential Radio Astronomy by CondonRansom2016: https://www.jstor.org/stable/j.ctv5vdcww


    Parameters:
        omgc: float [rad/s]
            Angular frequency of the pulsar.
        omgc_dot: float [rad/s^2]
            Angular frequency derivative of the pulsar.

    Returns:
        B_surf: float [G]
    """
    # P = 2 * np.pi / omgc
    # P_dot = 2 * np.pi * omgc_dot / omgc**2
    P = period_from_freq(omgc / (2 * np.pi))
    P_dot = P_dot_from_freq_freq_dot(omgc / (2 * np.pi), omgc_dot / (2 * np.pi))
    return 3.2e19 * (P * np.abs(P_dot)) ** (1 / 2)


def is_magnetar(B_surf) -> bool:
    """
    Check if a pulsar is a magnetar by checking if its surface magnetic field strength greater than the quantum critical threshold B_CRIT = 4.414e13 G. See e.g., https://academic.oup.com/mnras/article/378/1/159/1154016

    Parameters:
        B_surf: float [G]
            Surface magnetic field strength of the pulsar.
    """
    return B_surf > B_CRIT


def B_mf(omgc, tau):
    return 1 / (2 * omgc * tau)


def age_characteristic(f, fdot):
    """
    Calculate the characteristic age of a pulsar: age = f / (2 * f_dot).

    Returns:
        tau: float [s]
    """
    return np.abs(f / (2 * fdot))


def is_recycled_millisecond(f, f_dot) -> bool:
    """
    Check if a pulsar is regarded as a millisecond (recycled) pulsar.
    Criterion adopted from LeeEtAl2012: https://doi.org/10.1111/j.1365-2966.2012.21413.x


    Parameters:
        f: float [Hz]
            Frequency of the pulsar.
        f_dot: float [Hz/s]
            Frequency derivative of the pulsar.

    Returns:
        is_recycled: bool
            True if the pulsar is a recycled millisecond pulsar, False otherwise.
    """
    P_dot = P_dot_from_freq_freq_dot(f, f_dot)
    P = period_from_freq(f)
    return (P_dot / 1e-17) <= 3.23 * (P / (100 * MILLISECOND)) ** (-2.34)


def tau_eff_squared(taus_over_tauc, Qs, Qc, tau):
    tauc = tau * (1 + 1 / taus_over_tauc)
    tau_eff_sq_inv = (1 / taus_over_tauc**2 + Qs / Qc) / tauc**2

    return 1 / tau_eff_sq_inv


def timing_noise_strength(
    omgc, Qc, Qs, tau, taus_over_tauc, t_obs, n_obs, return_log10=True
):
    """
    Calculate the timing noise strength \sigma_{\\rm TN}^2 of a pulsar with the two-component model parameters.
    """
    tau_eff_sq = tau_eff_squared(taus_over_tauc, Qs, Qc, tau)
    common_factor = Qc * tau**2 / (3 * n_obs**3 * np.pi * tau_eff_sq * omgc**2)
    term1 = (n_obs**3 - 8) * t_obs**3
    term2 = (
        3
        * n_obs**2
        * (tau**2 - tau_eff_sq)
        * (
            n_obs
            * tau
            * (np.arctan(n_obs * tau / (2 * t_obs)) - np.arctan(tau / t_obs))
            - t_obs * (n_obs - 2)
        )
    )
    if return_log10:
        return np.log10(common_factor) + np.log10(term1 + term2)
    return common_factor * (term1 + term2)
