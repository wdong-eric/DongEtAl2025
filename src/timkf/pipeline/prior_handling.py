from typing import Dict
import bilby

LATEX_LABELS = {
    "EFAC": "EFAC",
    "EQUAD": "EQUAD",
    "phis_0": "$\\phi_{\\rm{s},0}$",
    "phic_0": "$\\phi_{\\rm{c},0}$",
    "omgc_0": "$\\Omega_{\\rm{c},0}$",
    "omgs_0": "$\\Omega_{\\rm{s},0}$",
    "omgc_dot": "$\\langle \\dot{\\Omega}_{\\rm{c}} \\rangle$",
    "lag": "$\\langle\\Omega_{\\rm{c}} - \\Omega_{\\rm{s}}\\rangle$",
    "Qc": "$Q_{\\rm{c}}$",
    "Qs": "$Q_{\\rm{s}}$",
    "ratio": "$\\tau_{\\rm{s}}/\\tau_{\\rm{c}}$",
    "xs": "$x_{\\rm{s}}$",
    "tau": "$\\tau$",
    "tauprime": "$\\tau'$",
}

UNIT_LABELS = {
    "EFAC": None,
    "EQUAD": "rad$^2$",  # unit is rad^2 because it is the variance of the phase
    "phis_0": "rad",
    "phic_0": "rad",
    "omgc_0": "rad s$^{-1}$",
    "omgs_0": "rad s$^{-1}$",
    "omgc_dot": "rad s$^{-2}$",
    "omgs_dot": "rad s$^{-2}$",
    "lag": "rad s$^{-1}$",
    "Qc": "rad$^2$ s$^{-3}$",
    "Qs": "rad$^2$ s$^{-3}$",
    "ratio": None,
    "xs": None,
    "tau": "s",
    "tauprime": "s",
}


def create_PriorDict(
    prior_dists: Dict[str, tuple], fixed_params: Dict[str, float] = None
):
    """
    Setup the priors for the parameters in the model.

    Args:
        prior_dists: Dictionary of parameter prior distributions.
            E.g., (distribution, low, high) or (delta, fixed_value), in the order of the blby intake
        fixed_params: Dictionary of fixed parameters, {"param_name": fixed_value}, which are the same as using DeltaFunction priors.

    Returns:
        PriorDict for Bilby sampling
    """
    # Define allowed distributions
    allowed_distributions = {
        "delta": ["delta", "diracdelta", "del"],
        "uniform": ["uniform", "unif", "u"],
        "loguniform": ["loguniform", "logunif", "logu"],
    }
    priors = bilby.core.prior.PriorDict()

    # Add fixed parameters as DeltaFunction priors
    if fixed_params is not None:
        for param, value in fixed_params.items():
            priors[param] = bilby.core.prior.DeltaFunction(
                value,
                name=param,
                latex_label=LATEX_LABELS[param],
                unit=UNIT_LABELS[param],
            )
    # Add prior distributions
    for param, tup in prior_dists.items():
        dist = tup[0]
        if dist in allowed_distributions["delta"]:
            priors[param] = bilby.core.prior.DeltaFunction(
                *tup[1:], latex_label=LATEX_LABELS[param], unit=UNIT_LABELS[param]
            )
        elif dist in allowed_distributions["loguniform"]:
            priors[param] = bilby.core.prior.LogUniform(
                *tup[1:],
                name=param,
                latex_label=LATEX_LABELS[param],
                unit=UNIT_LABELS[param],
            )
        elif dist in allowed_distributions["uniform"]:
            priors[param] = bilby.core.prior.Uniform(
                *tup[1:],
                name=param,
                latex_label=LATEX_LABELS[param],
                unit=UNIT_LABELS[param],
            )
        else:
            raise ValueError(
                f"Unsupported distribution '{dist}' for parameter '{param}'. "
                f"Supported distributions are: {allowed_distributions.keys()}"
            )

    return priors
