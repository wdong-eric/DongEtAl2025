# Common distribution functions for better reusability
import numpy as np


def log10_normal(x, mu_log10, sigma_log10):
    sigma_ln = sigma_log10 / np.log10(np.e)
    return (
        np.exp(-((np.log10(x) - mu_log10) ** 2) / (2 * sigma_log10**2))
        / np.sqrt(2 * np.pi)
        / x
        / sigma_ln
    )


def log_uniform(x, min_a, max_b):
    in_range = (x >= min_a) & (x <= max_b)
    return in_range / (x * (np.log(max_b) - np.log(min_a)))


def log10_uniform(x, min_log10, max_log10):
    in_range = (np.log10(x) >= min_log10) & (np.log10(x) <= max_log10)
    return in_range / (x * (max_log10 - min_log10) * np.log(10))


def log10norm_Log10unif_2MIXTURE(x, mu_log10, sigma_log10, min_log10, max_log10, pi1):
    """
    See page 522 in BDA3. The expression may be obtained by marginalising equation~(22.2) over the latent subpopulation status z_i (i = 1, ..., H subpopulation).

    =====
    Parameters:
    pi_0: float
        The mixture probability/proportion of the first component.
    """
    pis = [pi1, 1 - pi1]

    weighted_prob1 = pis[0] * log10_normal(x, mu_log10, sigma_log10)
    weighted_prob2 = pis[1] * log10_uniform(x, min_log10, max_log10)

    return weighted_prob1 + weighted_prob2


def log10norm_log10norm_2MIXTURE(
    x, mu1_log10, sigma1_log10, mu2_log10, sigma2_log10, pi1
):
    """
    See page 522 in BDA3. The expression may be obtained by marginalising equation~(22.2) over the latent subpopulation status z_i (i = 1, ..., H subpopulation).

    =====
    Parameters:
    pi_1: float
        The mixture probability/proportion of the first component.
    """
    pis = [pi1, 1 - pi1]

    weighted_prob1 = pis[0] * log10_normal(x, mu1_log10, sigma1_log10)
    weighted_prob2 = pis[1] * log10_normal(x, mu2_log10, sigma2_log10)

    return weighted_prob1 + weighted_prob2


def log10normTwo_log10unif_3MIXTURE(
    x,
    mu1_log10,
    sigma1_log10,
    mu2_log10,
    sigma2_log10,
    min_log10,
    max_log10,
    pi1,
    pi2,
):
    """
    See page 522 in BDA3. The expression may be obtained by marginalising equation~(22.2) over the latent subpopulation status z_i (i = 1, ..., H subpopulation).

    =====
    Parameters:
    pi_1,2: float
        The mixture probability/proportion of the first/second component.
    """
    pis = [pi1, pi2, 1 - pi1 - pi2]

    weighted_prob1 = pis[0] * log10_normal(x, mu1_log10, sigma1_log10)
    weighted_prob2 = pis[1] * log10_normal(x, mu2_log10, sigma2_log10)
    weighted_prob3 = pis[2] * log10_uniform(x, min_log10, max_log10)

    return weighted_prob1 + weighted_prob2 + weighted_prob3


def log10norm_log10unif_PreCategorised(
    x, cat, mu_log10, sigma_log10, min_log10, max_log10
):
    return (log10_normal(x, mu_log10, sigma_log10) ** cat) * log10_uniform(
        x, min_log10, max_log10
    ) ** (1 - cat)


def log10norm_powerlaw_corr_against_omgc_omgcdot(x, omgc, omgc_dot, C2, a, b, sigma_log10):
    """
    A base-10 lognormal hypermodel with the mu replaced by a power-law function of omega_c and omega_c_dot in log10-space:

    ..math::
        `\mu = \log_{10}(C_2 * \omega_c^a * \dot{\omega_c}^b)`

    This is to test the correlation between the parameter of interest and pulsar intrinsic parameters (which consists of omega_c and omega_c_dot).

    """
    mu_log10 = C2 + a * np.log10(omgc) + b * np.log10(np.abs(omgc_dot))

    return log10_normal(x, mu_log10, sigma_log10)


def log10norm_powerlaw_corr_against_omgc_omgcdot_Tobs(
    x, omgc, omgc_dot, Tobs, C2, a, b, gamma, sigma_log10
):
    mu_log10 = (
        C2
        + a * np.log10(omgc)
        + b * np.log10(np.abs(omgc_dot))
        + gamma * np.log10(Tobs)
    )

    return log10_normal(x, mu_log10, sigma_log10)
