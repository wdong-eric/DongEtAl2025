import numpy as np
import bilby
import inspect
from abc import ABC, abstractmethod
from .distribution_funcs import (
    log10_normal,
    log10norm_Log10unif_2MIXTURE,
    log10norm_log10norm_2MIXTURE,
    log10normTwo_log10unif_3MIXTURE,
    log10norm_log10unif_PreCategorised,
    log10norm_powerlaw_corr_against_omgc_omgcdot,
    log10norm_powerlaw_corr_against_omgc_omgcdot_Tobs,
)


class BaseHyperModel(ABC):
    def __init__(
        self,
        hp_prior_range: dict[str, tuple | list[tuple]],
        prior_latex_labels: dict[str, str] = None,
    ):
        self.hp_prior_range = hp_prior_range
        self.prior_latex_labels = prior_latex_labels or {
            key: key for key in self.hp_prior_range.keys()
        }

    def build_hyper_model(self):
        """Factory method to create bilby hypermodel"""
        return bilby.hyper.model.Model(self.build_hyper_prior_funcs())

    @abstractmethod
    def build_hyper_prior_funcs(self):
        raise NotImplementedError

    @abstractmethod
    def build_hp_priors(self):
        raise NotImplementedError

    @staticmethod
    def _create_hyper_prior_function(
        key: str,
        distribution_func: callable,
        dataset_keys: list[str] = None,
    ) -> callable:
        # in case of multiple keys from the dataset to be used
        dataset_keys = dataset_keys or [key]
        param_offset = len(dataset_keys)
        # skip the first few args, which is the data
        dist_args = inspect.getfullargspec(distribution_func).args[param_offset:]
        param_names = [f"{arg}_{key}" for arg in dist_args]

        hyper_prior_func_code = f"""
def hyper_prior_{key}(dataset, {", ".join(param_names)}):
    dataset_values = [dataset[dk] for dk in {dataset_keys}]
    return distribution_func(*dataset_values, {", ".join(param_names)})
"""
        exec(
            hyper_prior_func_code,
            {"distribution_func": distribution_func},
            locals(),
        )
        return eval(f"hyper_prior_{key}")

    def _extract_hp_prior_range(self, key: str, index: str = "") -> tuple:
        if isinstance(self.hp_prior_range[key], tuple):
            return self.hp_prior_range[key]

        if index:
            return self.hp_prior_range[key][int(index) - 1]

        flattened_range = np.array(
            [[a, b] for a, b in self.hp_prior_range[key]]
        ).flatten()
        return np.min(flattened_range), np.max(flattened_range)

    def _create_hp_prior_for_normal(self, key: str, index: str = "") -> dict:
        min_log10, max_log10 = np.log10(self._extract_hp_prior_range(key, index))

        prior_dict = {
            f"mu{index}_log10_{key}": bilby.core.prior.Uniform(
                min_log10,
                max_log10,
                name=f"mu{index}_log10_{key}",
                latex_label=r"$\mu_{\rm{%s}, {%s}}$"
                % (self.prior_latex_labels[key], index),
            ),
            f"sigma{index}_log10_{key}": bilby.core.prior.Uniform(
                0,
                1e1,
                name=f"sigma{index}_log10_{key}",
                latex_label=r"$\sigma_{\rm{%s}, {%s}}$"
                % (self.prior_latex_labels[key], index),
            ),
        }

        return prior_dict


class Log10normHyperModel(BaseHyperModel):
    """
    Parameters:
    -----------
        hp_prior_range: dict[str, tuple]
            Dictionary of the (linear scale) range of the prior of hyperparameters. Note that the key of the dict will be used as the subdirectory name of the output.
        prior_latex_labels: dict
            Dictionary of the latex labels of the hyperparameters. If none, the key of hp_prior_range will be used.
    """

    def build_hyper_prior_funcs(self):
        return [
            self._create_hyper_prior_function(key, log10_normal)
            for key in self.hp_prior_range.keys()
        ]

    def build_hp_priors(self):
        hp_priors = bilby.core.prior.PriorDict()

        for key in self.hp_prior_range.keys():
            hp_priors.update(self._create_hp_prior_for_normal(key))
        return hp_priors


class Log10normLog10unifHyperModel(BaseHyperModel):
    def build_hyper_prior_funcs(self):
        return [
            self._create_hyper_prior_function(key, log10norm_Log10unif_2MIXTURE)
            for key in self.hp_prior_range.keys()
        ]

    def build_hp_priors(self):
        def convert_min_max_to_diffConstraint(parameters):
            """
            Convert min_log10_{key} and max_log10_{key} to diff_log10_{key}.
            Modified according to the bilby official example.
            """
            converted_parameters = parameters.copy()
            for key in self.hp_prior_range.keys():
                converted_parameters[f"diff_log10_{key}"] = (
                    parameters[f"max_log10_{key}"] - parameters[f"min_log10_{key}"]
                )
            return converted_parameters

        hp_priors = bilby.core.prior.PriorDict(
            conversion_function=convert_min_max_to_diffConstraint
        )

        for key in self.hp_prior_range.keys():
            hp_priors.update(self._create_hp_prior_for_normal(key))

            min_log10, max_log10 = np.log10(self._extract_hp_prior_range(key))
            hp_priors.update(
                {
                    f"min_log10_{key}": bilby.core.prior.Uniform(
                        min_log10,
                        max_log10,
                        name=f"min_log10_{key}",
                        latex_label=r"$\min_{\rm{%s}}$" % self.prior_latex_labels[key],
                    ),
                    f"max_log10_{key}": bilby.core.prior.Uniform(
                        min_log10,
                        max_log10,
                        name=f"max_log10_{key}",
                        latex_label=r"$\max_{\rm{%s}}$" % self.prior_latex_labels[key],
                    ),
                    f"pi1_{key}": bilby.core.prior.DirichletElement(
                        order=0,
                        n_dimensions=2,
                        label=f"pi_{key}",
                    ),
                }
            )

            hp_priors[f"diff_log10_{key}"] = bilby.core.prior.Constraint(
                minimum=0,
                maximum=np.abs(max_log10 - min_log10),
            )

        return hp_priors


class Log10normLog10unifFixedBdryHyperModel(Log10normLog10unifHyperModel):
    def build_hp_priors(self):
        hp_priors = bilby.core.prior.PriorDict()

        for key in self.hp_prior_range.keys():
            hp_priors.update(self._create_hp_prior_for_normal(key))

            min_log10, max_log10 = np.log10(self._extract_hp_prior_range(key))
            hp_priors.update(
                {
                    f"min_log10_{key}": bilby.core.prior.DeltaFunction(
                        min_log10,
                        name=f"min_log10_{key}",
                        latex_label=r"$\min_{\rm{%s}}$" % self.prior_latex_labels[key],
                    ),
                    f"max_log10_{key}": bilby.core.prior.DeltaFunction(
                        max_log10,
                        name=f"max_log10_{key}",
                        latex_label=r"$\max_{\rm{%s}}$" % self.prior_latex_labels[key],
                    ),
                    f"pi1_{key}": bilby.core.prior.DirichletElement(
                        order=0,
                        n_dimensions=2,
                        label=f"pi_{key}",
                    ),
                }
            )

        return hp_priors


class Log10normLog10unifFixedOneBdryHyperModel(Log10normLog10unifHyperModel):
    def build_hp_priors(self):
        hp_priors = bilby.core.prior.PriorDict()

        for key in self.hp_prior_range.keys():
            hp_priors.update(self._create_hp_prior_for_normal(key))

            min_log10, max_log10 = np.log10(self._extract_hp_prior_range(key))
            if key == "tau" or key == "tauprime":
                hp_priors.update(
                    {
                        f"min_log10_{key}": bilby.core.prior.Uniform(
                            min_log10,
                            max_log10,
                            name=f"min_log10_{key}",
                            latex_label=r"$\min_{\rm{%s}}$"
                            % self.prior_latex_labels[key],
                        ),
                        f"max_log10_{key}": bilby.core.prior.DeltaFunction(
                            max_log10,
                            name=f"max_log10_{key}",
                            latex_label=r"$\max_{\rm{%s}}$"
                            % self.prior_latex_labels[key],
                        ),
                    }
                )
            else:
                hp_priors.update(
                    {
                        f"min_log10_{key}": bilby.core.prior.DeltaFunction(
                            min_log10,
                            name=f"min_log10_{key}",
                            latex_label=r"$\min_{\rm{%s}}$"
                            % self.prior_latex_labels[key],
                        ),
                        f"max_log10_{key}": bilby.core.prior.Uniform(
                            min_log10,
                            max_log10,
                            name=f"max_log10_{key}",
                            latex_label=r"$\max_{\rm{%s}}$"
                            % self.prior_latex_labels[key],
                        ),
                    }
                )
            hp_priors.update(
                {
                    f"pi1_{key}": bilby.core.prior.DirichletElement(
                        order=0,
                        n_dimensions=2,
                        label=f"pi_{key}",
                    ),
                }
            )

        return hp_priors


class Log10normLog10unifFixedBdryPreCategorisedHyperModel(BaseHyperModel):
    def build_hyper_prior_funcs(self):
        return [
            self._create_hyper_prior_function(
                key,
                log10norm_log10unif_PreCategorised,
                [key, f"cat_{key}"],
            )
            for key in self.hp_prior_range.keys()
        ]

    def build_hp_priors(self):
        hp_priors = bilby.core.prior.PriorDict()

        for key in self.hp_prior_range.keys():
            hp_priors.update(self._create_hp_prior_for_normal(key))

            min_log10, max_log10 = np.log10(self._extract_hp_prior_range(key))
            hp_priors.update(
                {
                    f"min_log10_{key}": bilby.core.prior.DeltaFunction(
                        min_log10,
                        name=f"min_log10_{key}",
                        latex_label=r"$\min_{\rm{%s}}$" % self.prior_latex_labels[key],
                    ),
                    f"max_log10_{key}": bilby.core.prior.DeltaFunction(
                        max_log10,
                        name=f"max_log10_{key}",
                        latex_label=r"$\max_{\rm{%s}}$" % self.prior_latex_labels[key],
                    ),
                }
            )

        return hp_priors


class Log10normLog10unifFixedOneBdryPreCategorisedHyperModel(
    Log10normLog10unifFixedBdryPreCategorisedHyperModel
):
    def build_hp_priors(self):
        hp_priors = bilby.core.prior.PriorDict()

        for key in self.hp_prior_range.keys():
            hp_priors.update(self._create_hp_prior_for_normal(key))

            min_log10, max_log10 = np.log10(self._extract_hp_prior_range(key))
            if key == "tau" or key == "tauprime":
                hp_priors.update(
                    {
                        f"min_log10_{key}": bilby.core.prior.Uniform(
                            min_log10,
                            max_log10,
                            name=f"min_log10_{key}",
                            latex_label=r"$\min_{\rm{%s}}$"
                            % self.prior_latex_labels[key],
                        ),
                        f"max_log10_{key}": bilby.core.prior.DeltaFunction(
                            max_log10,
                            name=f"max_log10_{key}",
                            latex_label=r"$\max_{\rm{%s}}$"
                            % self.prior_latex_labels[key],
                        ),
                    }
                )
            else:
                hp_priors.update(
                    {
                        f"min_log10_{key}": bilby.core.prior.DeltaFunction(
                            min_log10,
                            name=f"min_log10_{key}",
                            latex_label=r"$\min_{\rm{%s}}$"
                            % self.prior_latex_labels[key],
                        ),
                        f"max_log10_{key}": bilby.core.prior.Uniform(
                            min_log10,
                            max_log10,
                            name=f"max_log10_{key}",
                            latex_label=r"$\max_{\rm{%s}}$"
                            % self.prior_latex_labels[key],
                        ),
                    }
                )

        return hp_priors


class Log10normLog10unifPreCategorisedHyperModel(
    Log10normLog10unifFixedBdryPreCategorisedHyperModel
):
    def build_hp_priors(self):
        def convert_min_max_to_diffConstraint(parameters):
            """
            Convert min_log10_{key} and max_log10_{key} to diff_log10_{key}.
            Modified according to the bilby official example.
            """
            converted_parameters = parameters.copy()
            for key in self.hp_prior_range.keys():
                converted_parameters[f"diff_log10_{key}"] = (
                    parameters[f"max_log10_{key}"] - parameters[f"min_log10_{key}"]
                )
            return converted_parameters

        hp_priors = bilby.core.prior.PriorDict(
            conversion_function=convert_min_max_to_diffConstraint
        )

        for key in self.hp_prior_range.keys():
            hp_priors.update(self._create_hp_prior_for_normal(key))

            min_log10, max_log10 = np.log10(self._extract_hp_prior_range(key))
            hp_priors[f"min_log10_{key}"] = bilby.core.prior.Uniform(
                min_log10,
                max_log10,
                name=f"min_log10_{key}",
                latex_label=r"$\\min_{\rm{%s}}$" % self.prior_latex_labels[key],
            )

            hp_priors[f"max_log10_{key}"] = bilby.core.prior.Uniform(
                min_log10,
                max_log10,
                name=f"max_log10_{key}",
                latex_label=r"$\\max_{\rm{%s}}$" % self.prior_latex_labels[key],
            )

            hp_priors[f"diff_log10_{key}"] = bilby.core.prior.Constraint(
                minimum=0,
                maximum=np.abs(max_log10 - min_log10),
            )

        return hp_priors


class TwoLog10normHyperModel(BaseHyperModel):
    def build_hyper_prior_funcs(self):
        return [
            self._create_hyper_prior_function(key, log10norm_log10norm_2MIXTURE)
            for key in self.hp_prior_range.keys()
        ]

    def build_hp_priors(self):
        hp_priors = bilby.core.prior.PriorDict()

        for key in self.hp_prior_range.keys():
            hp_priors.update(self._create_hp_prior_for_normal(key, index="1"))
            hp_priors.update(self._create_hp_prior_for_normal(key, index="2"))
            hp_priors.update(
                {
                    f"pi1_{key}": bilby.core.prior.DirichletElement(
                        order=0,
                        n_dimensions=2,
                        label=f"pi_{key}",
                    ),
                }
            )

        return hp_priors


class TwoLog10normLog10unifFixedOneBdryHyperModel(BaseHyperModel):
    def build_hyper_prior_funcs(self):
        return [
            self._create_hyper_prior_function(key, log10normTwo_log10unif_3MIXTURE)
            for key in self.hp_prior_range.keys()
        ]

    def build_hp_priors(self):
        hp_priors = bilby.core.prior.PriorDict()

        for key in self.hp_prior_range.keys():
            hp_priors.update(self._create_hp_prior_for_normal(key, index="1"))
            hp_priors.update(self._create_hp_prior_for_normal(key, index="2"))

            min_log10, max_log10 = np.log10(self._extract_hp_prior_range(key))
            if key == "tau" or key == "tauprime":
                hp_priors.update(
                    {
                        f"min_log10_{key}": bilby.core.prior.Uniform(
                            min_log10,
                            max_log10,
                            name=f"min_log10_{key}",
                            latex_label=r"$\min_{\rm{%s}}$"
                            % self.prior_latex_labels[key],
                        ),
                        f"max_log10_{key}": bilby.core.prior.DeltaFunction(
                            max_log10,
                            name=f"max_log10_{key}",
                            latex_label=r"$\max_{\rm{%s}}$"
                            % self.prior_latex_labels[key],
                        ),
                    }
                )
            else:
                hp_priors.update(
                    {
                        f"min_log10_{key}": bilby.core.prior.DeltaFunction(
                            min_log10,
                            name=f"min_log10_{key}",
                            latex_label=r"$\min_{\rm{%s}}$"
                            % self.prior_latex_labels[key],
                        ),
                        f"max_log10_{key}": bilby.core.prior.Uniform(
                            min_log10,
                            max_log10,
                            name=f"max_log10_{key}",
                            latex_label=r"$\max_{\rm{%s}}$"
                            % self.prior_latex_labels[key],
                        ),
                    }
                )
            hp_priors.update(
                {
                    f"pi1_{key}": bilby.core.prior.DirichletElement(
                        order=0,
                        n_dimensions=3,
                        label=f"pi_{key}",
                    ),
                    f"pi2_{key}": bilby.core.prior.DirichletElement(
                        order=1,
                        n_dimensions=3,
                        label=f"pi_{key}",
                    ),
                }
            )

        return hp_priors


class PowerLawCorrAgainstOmgcOmgcdotHyperModel(BaseHyperModel):
    def build_hyper_prior_funcs(self):
        return [
            self._create_hyper_prior_function(
                key,
                log10norm_powerlaw_corr_against_omgc_omgcdot,
                [key, "omgc_0", "omgc_dot"],
            )
            for key in self.hp_prior_range.keys()
        ]

    def build_hp_priors(self):
        hp_priors = bilby.core.prior.PriorDict()

        for key in self.hp_prior_range.keys():
            C2_min, C2_max = np.log10(self._extract_hp_prior_range(key))
            hp_priors.update(
                {
                    f"C2_{key}": bilby.core.prior.Uniform(
                        C2_min - 30,
                        C2_max + 30,
                        name=f"C2_{key}",
                        latex_label=r"$C_{2,\rm{%s}}$" % self.prior_latex_labels[key],
                    ),
                }
            )
            hp_priors.update(
                {
                    f"a_{key}": bilby.core.prior.Uniform(
                        -10,
                        10,
                        name=f"a_{key}",
                        latex_label=r"$a_{\rm{%s}}$" % self.prior_latex_labels[key],
                    ),
                    f"b_{key}": bilby.core.prior.Uniform(
                        -10,
                        10,
                        name=f"b_{key}",
                        latex_label=r"$b_{\rm{%s}}$" % self.prior_latex_labels[key],
                    ),
                }
            )
            hp_priors.update(
                {
                    f"sigma_log10_{key}": bilby.core.prior.Uniform(
                        0,
                        1e1,
                        name=f"sigma_log10_{key}",
                        latex_label=r"$\sigma_{\rm{%s}}$"
                        % self.prior_latex_labels[key],
                    ),
                }
            )
        return hp_priors


class PowerLawCorrAgainstOmgcOmgcdotTobsHyperModel(BaseHyperModel):
    def build_hyper_prior_funcs(self):
        return [
            self._create_hyper_prior_function(
                key,
                log10norm_powerlaw_corr_against_omgc_omgcdot_Tobs,
                [key, "omgc_0", "omgc_dot", "Tobs"],
            )
            for key in self.hp_prior_range.keys()
        ]

    def build_hp_priors(self):
        hp_priors = bilby.core.prior.PriorDict()

        for key in self.hp_prior_range.keys():
            C2_min, C2_max = -30, 60  # hard-coded
            if key == "Qs":
                C2_min, C2_max = 0, 90
            hp_priors.update(
                {
                    f"C2_{key}": bilby.core.prior.Uniform(
                        C2_min,
                        C2_max,
                        name=f"C2_{key}",
                        latex_label=r"$C_{2,\rm{%s}}$" % self.prior_latex_labels[key],
                    ),
                }
            )
            hp_priors.update(
                {
                    f"a_{key}": bilby.core.prior.Uniform(
                        -10,
                        10,
                        name=f"a_{key}",
                        latex_label=r"$a_{\rm{%s}}$" % self.prior_latex_labels[key],
                    ),
                    f"b_{key}": bilby.core.prior.Uniform(
                        -10,
                        10,
                        name=f"b_{key}",
                        latex_label=r"$b_{\rm{%s}}$" % self.prior_latex_labels[key],
                    ),
                    f"gamma_{key}": bilby.core.prior.Uniform(
                        -10,
                        10,
                        name=f"gamma_{key}",
                        latex_label=r"$\gamma_{\rm{%s}}$"
                        % self.prior_latex_labels[key],
                    ),
                }
            )
            hp_priors.update(
                {
                    f"sigma_log10_{key}": bilby.core.prior.Uniform(
                        0,
                        1e1,
                        name=f"sigma_log10_{key}",
                        latex_label=r"$\sigma_{\rm{%s}}$"
                        % self.prior_latex_labels[key],
                    ),
                }
            )
        return hp_priors
