import os
import bilby
from bilby.core.prior import PriorDict
import numpy as np
from typing import Literal
from configparser import ConfigParser
from ..core import KalmanFilterStandard
from ..core.sampling import BaseKFLikelihood
from .prior_handling import create_PriorDict


class AnalysisRunner:
    def __init__(
        self,
        src_config: ConfigParser,
    ):
        self.sampler_config = src_config["sampler_config"]

    def setup_priors(
        self,
        prior_dict_input: dict,
        tempo_psr: Literal["libstempo.tempopulsar"],
        omgc_0_ref,
        omgc_dot_ref,
        fixed_params: dict[float] = None,
    ) -> PriorDict:
        """
        Configure bilby prior dictionary.
        TODO: add validation for parameters, e.g. check if there are enough ones
        """
        if fixed_params is None:
            fixed_params = {}
        # == Set up priors ==
        assert (set(prior_dict_input.keys()) & set(fixed_params.keys())) == set(), (
            "Priors and fixed parameters have overlapping keys. Please check the input."
        )
        # Also sample over omgc_0 and omgc_dot if not provided
        STD_FACTOR = 1e3
        defined_params_union = set(prior_dict_input.keys()) | set(fixed_params.keys())
        if "omgc_0" not in defined_params_union:
            omgc0_err = tempo_psr["F0"].err * 2 * np.pi * STD_FACTOR
            omgc_0_low, omgc_0_high = (
                np.float64(omgc_0_ref - omgc0_err),
                np.float64(omgc_0_ref + omgc0_err),
            )
            prior_dict_input.update({"omgc_0": ("u", omgc_0_low, omgc_0_high)})
        if "omgc_dot" not in defined_params_union:
            omgc_dot_err = tempo_psr["F1"].err * 2 * np.pi * STD_FACTOR
            omgc_dot_low, omgc_dot_high = (
                np.float64(omgc_dot_ref - omgc_dot_err),
                np.float64(omgc_dot_ref + omgc_dot_err),
            )
            prior_dict_input.update({"omgc_dot": ("u", omgc_dot_low, omgc_dot_high)})
        # == Default fixed parameters if not provided ==
        if "EFAC" not in defined_params_union:
            # Error FACtor, close to 1 if timing model is well and noise is white
            fixed_params["EFAC"] = 1.0
        if "EQUAD" not in defined_params_union:
            # Error QUADrature, 0 if no extra white noise
            fixed_params["EQUAD"] = 0.0

        self.priors = create_PriorDict(prior_dict_input, fixed_params)

        return self.priors

    def setup(self, model):
        """Initialize the kalman filter (KF) for input model, sampling parameters and KF likelihood for bilby usage."""
        # construct the kalman filter
        self.kf = KalmanFilterStandard(model)
        # contruct the likelihood
        sampling_params = {f"{key}": None for key in self.priors.keys()}
        print(f"{sampling_params=}")
        self.likelihood = BaseKFLikelihood(self.kf, sampling_params)

    def run_sampling(self, sampler_outdir, sampler_tag) -> bilby.result.Result:
        """Execute the sampling process with configured settings."""
        assert self.likelihood is not None and self.priors is not None, (
            "Likelihood and Priors not set up. Call setup() first."
        )
        ncores = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
        print(f"{ncores} CPU cores available. Using all {ncores} for parallelisation\n")

        result = bilby.run_sampler(
            self.likelihood,
            self.priors,
            sampler=self.sampler_config.get("sampler", "dynesty"),
            sample="rwalk",
            walks=self.sampler_config.getint("Nwalks"),
            npoints=self.sampler_config.getint(
                "Npoints", len(self.priors) * 50
            ),  # let's default 50 points per parameter
            resume=self.sampler_config.getboolean("resume_run"),
            outdir=sampler_outdir,
            label=sampler_tag,
            check_point_plot=False,
            seed=self.sampler_config.getint("seed"),
            npool=ncores,
        )

        return result
