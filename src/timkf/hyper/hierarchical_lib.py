import os
import bilby
import pandas as pd
import numpy as np
import inspect
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Literal
from corner import corner

from .hyper_model import BaseHyperModel
from ..misc import overlay_corner2, gen_load_bilby_results, PSRJ_name


class HierarchicalPsrPop:
    """
    Class to perform hierarchical analysis on the pulsar population data.

    Parameters:
    -----------
    filtered_data_path_parnt: str
        Parent directory of the filtered data.
        The output of the hierarchical analysis will be saved in <filtered_data_path_parnt>/hrchy/

    hp_model: e.g. HyperModelLog10Norm | HyperModelLog10NormAndLogUniform
        Hyperparameter model to be used. Only two models are supported:
        - HyperModelLog10Norm: Log10-normal distribution for all hyperparameters.
        - HyperModelLog10NormAndLogUniform: a combination of Log10-normal and Log-uniform distributions.

        Both models require the following attributes:
            hp_prior_range: dict[str, tuple]
                Dictionary of the (linear scale) range of the prior of hyperparameters. The key of the dict will be used as the subdirectory name of the output.
            prior_latex_labels: dict
                Dictionary of the latex labels of the hyperparameters. If none, the key of hp_prior_range will be used.
        See timkf.hyper.hyper_model.py for details.

    restrict_to_psr: list[PSRJ_name]
        List of pulsar names to restrict the analysis to. If None, all pulsars in the filtered data will be used.
    """

    def __init__(
        self,
        filtered_data_path_parnt: str,
        hp_model: BaseHyperModel,
        categorical_info: pd.DataFrame = None,
        sampler="dynesty",
    ):
        self.sampler = sampler
        self.filtered_data_path_parnt = filtered_data_path_parnt

        self.hp_model = hp_model
        self.hp_prior_range = hp_model.hp_prior_range
        self.categorical_info = categorical_info

        self.hrchy_outdir = os.path.join(
            self.filtered_data_path_parnt,
            "hrchy",
            "-".join(sorted(list(self.hp_prior_range.keys()))),
        )

    def __call__(
        self,
        restrict_to_psr: list[PSRJ_name] = None,
        tag="",
        categorical_info: pd.DataFrame = None,
        Tobs_info: pd.DataFrame = None,
    ):
        if "categorised" in self.hp_model.__class__.__name__.lower():
            assert categorical_info is not None, (
                "categorical_info must be provided for categorical hypermodel"
            )
            self.categorical_info = categorical_info
        self.Tobs_info = Tobs_info

        pop_data, pop_log_evidences = self.prepare_sample_and_evidence_for_hrchy(
            restrict_to_psr
        )
        result = self.run_hierarchical_sampling(pop_data, pop_log_evidences, tag)
        return result

    def prepare_sample_and_evidence_for_hrchy(
        self, restrict_to_psr: list[PSRJ_name] = None, as_dict=False
    ) -> tuple[list[pd.DataFrame], list[Literal["log_evidence"]]]:
        psr_names, pop_samples, evidences = [], [], []
        for psr_name, bilby_result in gen_load_bilby_results(
            self.filtered_data_path_parnt, restrict_to=restrict_to_psr
        ):
            # we only need priors/log_priors of params that is used for hierarchical analysis;
            df_samples: pd.DataFrame = bilby_result.posterior
            df_samples.pop("log_prior")
            df_samples["prior"] = 1
            if self.categorical_info is not None:
                for cat in self.categorical_info.columns:
                    df_samples[cat] = self.categorical_info.loc[psr_name, cat]
            if self.Tobs_info is not None:
                df_samples["Tobs"] = self.Tobs_info.loc[psr_name, "Tobs"]

            for key in self.hp_prior_range.keys():
                assert key in df_samples.columns, f"{key} not in {df_samples.columns}"
                df_samples["prior"] *= bilby_result.priors[key].prob(df_samples[key])

            psr_names.append(psr_name)
            pop_samples.append(df_samples)
            evidences.append(bilby_result.log_evidence)

        if as_dict:
            pop_samples = dict(zip(psr_names, pop_samples))
            evidences = dict(zip(psr_names, evidences))

        return pop_samples, evidences

    def hyper_param_likelihood(self, pop_data, pop_log_evidences):
        hyper_prior_model = self.hp_model.build_hyper_model()

        hp_likelihood = bilby.hyper.likelihood.HyperparameterLikelihood(
            posteriors=pop_data,
            hyper_prior=hyper_prior_model,
            log_evidences=pop_log_evidences,
        )

        return hp_likelihood

    def run_hierarchical_sampling(self, pop_data, pop_log_evidences, tag=""):
        hp_likelihood = self.hyper_param_likelihood(pop_data, pop_log_evidences)

        hp_priors = self.hp_model.build_hp_priors()

        result = bilby.run_sampler(
            likelihood=hp_likelihood,
            priors=hp_priors,
            sampler=self.sampler,
            nlive=max(50 * len(hp_priors), 500),  # 50 * ndim
            outdir=f"{self.hrchy_outdir}/{tag}",
            label=f"{tag}",
            npool=int(os.getenv("SLURM_CPUS_PER_TASK", 1)),
            resume=True,
        )
        return result


class HrchyPopReweightedPosterior:
    def __init__(
        self,
        hrchy_pop: HierarchicalPsrPop,
        hpp_tag: str,
        reweigh_psrs: list[PSRJ_name] = None,
    ):
        self.hrchy_pop = hrchy_pop
        self.hpp_tag = hpp_tag
        self.reweigh_psrs = reweigh_psrs

        self.hrchy_outdir_tagged = os.path.join(self.hrchy_pop.hrchy_outdir, hpp_tag)

        # check if the result json exists in the tagged directory
        result_jsons = [
            file
            for file in os.listdir(self.hrchy_outdir_tagged)
            if file.endswith(".json")
        ]
        if not result_jsons:
            raise FileNotFoundError(
                f"No result json found in {self.hrchy_outdir_tagged}"
            )
        assert len(result_jsons) == 1, (
            f"More than one result json found in {self.hrchy_outdir_tagged}"
        )
        # read in the hrchy result
        self.hrchy_result = bilby.result.read_in_result(
            os.path.join(self.hrchy_outdir_tagged, result_jsons[0])
        )
        self.hyper_samples = self.hrchy_result.posterior.copy()
        self.hyper_samples.pop("log_prior")
        self.hyper_samples.pop("log_likelihood")
        # store the posterior samples of the pulsars
        self.posterior_samples, _ = (
            self.hrchy_pop.prepare_sample_and_evidence_for_hrchy(
                self.reweigh_psrs, as_dict=True
            )
        )

        # generate the weights dict for each pulsar
        self.weights = self.get_weights_for_psrs()

    def prior_importance_ratio(
        self,
        posterior_samples_psr_i: pd.DataFrame,
        hyper_samples: pd.DataFrame,
    ) -> np.ndarray:
        """
        The ratios [list], contains the prior importance ratio for each pulsar [self.reweigh_psrs].

        Returns:
        --------
        np.array: 2D array of shape (n_posterior_samples_p, n_hyper_samples)
        """
        bilby_hyper_model = self.hrchy_pop.hp_model.build_hyper_prior_funcs()

        pop_informed_priors = np.ones(len(hyper_samples))
        for pip_function in bilby_hyper_model:
            # get the kwargs of the pip function
            func_params = inspect.signature(pip_function).parameters
            func_specific_sample_dict = {
                k: hyper_samples[k].values
                for k in hyper_samples.columns
                if k in func_params
            }

            # convert df to dict of array to ensure broadcasting inside pip_function
            posterior_sample = posterior_samples_psr_i.to_dict(orient="list")
            for key in posterior_sample.keys():
                posterior_sample[key] = np.array(posterior_sample[key])[:, np.newaxis]
            # don't use pop_informed_priors*=, as 1D array cannot be broadcasted to be the output array; a new array needs to be created
            pop_informed_priors = pop_informed_priors * pip_function(
                posterior_sample, **func_specific_sample_dict
            )
        # reshape posterior_samples_psr_i["prior"] to (n, 1) to ensure broadcast
        importance_ratios = (
            pop_informed_priors / posterior_samples_psr_i["prior"].values[:, np.newaxis]
        )

        return importance_ratios

    def weight_marginalising_over_hyperparams(
        self, posterior_samples_psr_i: pd.DataFrame
    ) -> np.array:
        importance_ratios = self.prior_importance_ratio(
            posterior_samples_psr_i, self.hyper_samples
        )
        importance_ratios /= np.sum(importance_ratios, axis=0)
        importance_ratios /= len(importance_ratios)

        return np.sum(importance_ratios, axis=1) / len(self.hyper_samples)

    def get_weights_for_psrs(self) -> dict[PSRJ_name, np.array]:
        """
        Returns:
        --------
        list[np.array]: list of weights for each pulsar in self.reweigh_psrs
        """
        weights_csv = os.path.join(self.hrchy_outdir_tagged, "weights.csv")
        if os.path.exists(weights_csv):
            # read in the weights from the csv file
            df = pd.read_csv(weights_csv, index_col=0, header=None)
            weights = {
                df.index[i]: df.iloc[i].values[df.iloc[i].notna()]
                for i in range(len(df.index))
            }
            return weights

        weights = {}
        for psr_name in tqdm(
            self.reweigh_psrs,
            desc="Calculating weights for each pulsar",
        ):
            weights[psr_name] = self.weight_marginalising_over_hyperparams(
                self.posterior_samples[psr_name]
            )
        # save the weights to a csv file
        df = pd.DataFrame.from_dict(weights, orient="index")
        df.to_csv(
            os.path.join(self.hrchy_outdir_tagged, "weights.csv"),
            index=True,
            header=False,
        )
        return weights

    def reweight_posteriors_hist(
        self, overlay: bool = False, restrict_to_psr: str = None, **corner_kwargs
    ):
        if restrict_to_psr is None:
            restrict_to_psr = self.reweigh_psrs
        if overlay:
            fig = overlay_corner2(
                self.hrchy_pop.filtered_data_path_parnt,
                restrict_to_psr=restrict_to_psr,
                weights=self.weights,
                **corner_kwargs,
            )
            fig.suptitle("Posteriors reweighted using population-informed priors")
            return fig

        for psr, wts in tqdm(
            self.weights.items(), desc="Reweight posteriors histogram for each pulsar"
        ):
            if psr != restrict_to_psr:
                continue
            fig = overlay_corner2(
                self.hrchy_pop.filtered_data_path_parnt,
                restrict_to_psr=[psr],
                weights=wts / np.sum(wts),
                show_titles=True,
                quantiles=[0.16, 0.5, 0.84],
                **corner_kwargs,
            )
            fig.suptitle(
                f"{psr} posteriors reweighted using population-informed priors"
            )
            plt.show()

    def resample_weighted_posteriors(self):
        for psr, wts in tqdm(
            self.weights.items(),
            desc="Resampling weighted posteriors for each pulsar",
        ):
            resampled_posterior = self.posterior_samples[psr].sample(
                n=10 * len(self.posterior_samples[psr]),
                replace=True,
                weights=wts,
            )
            resampled_posterior.pop("log_likelihood")
            resampled_posterior.pop("prior")
            resampled_posterior.pop("phic_0")
            resampled_posterior.pop("phis_0")
            fig = corner(
                resampled_posterior,
                range=[0.99] * len(resampled_posterior.columns),
                axes_scale=["log"] * 4 + ["linear"] * 4 + ["log"],
                show_titles=True,
                title_fmt=".2e",
                quantiles=[0.16, 0.5, 0.84],
                levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4.5)),
            )
            fig.suptitle(f"{psr} posteriors resampled using population-informed priors")
            plt.show()
        return fig

    @staticmethod
    def effective_sample_size(weights):
        """
        Calculate the effective sample size.
        The effective sample size is defined as:
            ESS = 1 / \sum_{s=1}^{N} w_s^2
        where w_s are the normalised weights; that is, w_s = w_s / \sum_{s=1}^{N} w_s
        """
        return np.sum(weights) ** 2 / np.sum(weights**2)

    def get_effective_sample_sizes(self):
        """
        Calculate the effective sample size for each pulsar.
        The effective sample size is defined as:
            ESS = 1 / \sum_{s=1}^{N} w_s^2
        where w_s are the normalised weights; that is, w_s = w_s / \sum_{s=1}^{N} w_s

        Returns:
            ess: dict[PSRJ_name, float]
                effective sample size for each pulsar
        """
        ess = {}
        for psr, wts in self.weights.items():
            ess[psr] = self.effective_sample_size(wts)
        return ess
