#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
from pathlib import Path

from timkf.core import ModelConfig, TwoComponentPhaseModel, WTNPhaseModel
from timkf.pipeline import PulsarDataLoader, PipelineCLI, AnalysisRunner
from timkf.misc import plot_phase_residuals, make_cornerplot

print("Importing modules completed")
print("The package path contains:", __package__)
print("Beginning program")


def analyse_twocomp(
    times,
    phases,
    meas_cov,
    omgc_0_par,
    omgc_dot_par,
    src_config,
    model_config: ModelConfig,
    sampler_outdir,
    sampler_tag,
    tempo_psr,
):
    # initial data illustration - without parameter estimation results
    plot_phase_residuals(
        sampler_outdir,
        times,
        phases,
        meas_cov,
        omgc_0_par,
        omgc_dot_par,
        result=None,
        tempo_psr=tempo_psr,
    )
    data = phases
    print(f"{meas_cov = }")
    if model_config.IS_RESIDUAL:
        data = tempo_psr.phaseresiduals() * (2 * mp.pi)

    # ============================================
    # == Set up priors ==
    prior_dict_input: dict = eval(src_config["sampler_config"]["prior_dists"])
    assert not ("Qs" in prior_dict_input and "tauprime" in prior_dict_input), (
        "Cannot set both Qs and tauprime in the prior dictionary. Please choose one."
    )
    if "Qs" not in prior_dict_input and "tauprime" not in prior_dict_input:
        prior_dict_input["Qs"] = prior_dict_input["Qc"]
    if "ratio" not in prior_dict_input:
        prior_dict_input.update({"xs": ("u", 0.0, 1.0)})
    # Set fixed parameters based on initial phase measurement
    fixed_params = {"phic_0": np.float64(data[0]), "phis_0": np.float64(data[0])}
    # Configure bilby prior dictionary
    analrunner = AnalysisRunner(src_config)
    analrunner.setup_priors(
        prior_dict_input, tempo_psr, omgc_0_par, omgc_dot_par, fixed_params=fixed_params
    )

    # ============================================
    # == Create the model ==
    meas_mat = mp.matrix([[1.0, 0.0, 0.0, 0.0]])
    model = TwoComponentPhaseModel(
        model_config,
        times,
        data,
        meas_cov,
        meas_mat,
    )
    analrunner.setup(model)
    result = analrunner.run_sampling(sampler_outdir, sampler_tag)

    plot_phase_residuals(
        sampler_outdir,
        times,
        phases,
        meas_cov,
        omgc_0_par,
        omgc_dot_par,
        result=result,
        tempo_psr=tempo_psr,
    )

    truths = []
    for col in result.search_parameter_keys:
        if col in ["omgc_0", "omgc_dot"]:
            truths.append(omgc_dot_par if col == "omgc_dot" else omgc_0_par)
        else:
            truths.append(None)

    _, corner_kwargs = make_cornerplot(
        result, save_figname=f"{sampler_outdir}/{sampler_tag}_corner.png", truths=truths
    )

    return result, corner_kwargs


def analyse_onecompWTN(
    times,
    phases,
    meas_cov,
    omgc_0_par,
    omgc_dot_par,
    src_config,
    model_config,
    sampler_outdir,
    sampler_tag,
    tempo_psr,
):
    # == Set up priors ==
    prior_dict_input: dict = eval(src_config["sampler_config"]["wtn_prior_dists"])
    nullH_model = WTNPhaseModel(model_config, times, phases, meas_cov)
    # Set fixed parameters based on initial phase measurement
    fixed_params = {"phic_0": np.float64(phases[0])}
    # Configure bilby prior dictionary
    analrunner = AnalysisRunner(src_config)
    analrunner.setup_priors(
        prior_dict_input, tempo_psr, omgc_0_par, omgc_dot_par, fixed_params=fixed_params
    )
    analrunner.setup(nullH_model)
    nullH_result = analrunner.run_sampling(sampler_outdir, sampler_tag)

    truths = []
    for col in nullH_result.search_parameter_keys:
        if col in ["omgc_0", "omgc_dot"]:
            truths.append(omgc_dot_par if col == "omgc_dot" else omgc_0_par)
        elif col == "EFAC":
            try:
                truths.append(tempo_psr["TNGlobalEF"].val ** 2)
            except KeyError:
                truths.append(None)
        elif col == "EQUAD":
            try:
                truths.append(omgc_0_par * 10 ** (2 * tempo_psr["TNGLobalEQ"].val))
            except KeyError:
                truths.append(None)
        else:
            truths.append(None)
    _ = make_cornerplot(
        nullH_result,
        save_figname=f"{sampler_outdir}/{sampler_tag}_WTNnullH_corner.png",
        truths=truths,
    )
    plt.savefig(f"{sampler_outdir}/{sampler_tag}_WTNnullH_corner.png")
    plt.close()


def main():
    pipecli = PipelineCLI()
    prsrargs = pipecli.args
    src_config = pipecli.SRC_CONFIG

    (times, phases, R_phase), (omgc_0_par, omgc_dot_par), (tempo_psr, tempo_res) = (
        PulsarDataLoader(prsrargs.nume_impl).load(prsrargs.parfile, prsrargs.timfile)
    )
    # only run if not a glitching pulsar
    if [p for p in tempo_psr.pars(which="set") if p.startswith("GLEP")]:
        if not prsrargs.run_glitch_psr:
            raise TypeError(
                f"{tempo_psr.name} is a glitching pulsar. Please use the glitching pipeline instead. Skipping..."
            )
    outdirectory = Path(prsrargs.out_directory)
    outdirectory.mkdir(parents=True, exist_ok=True)

    # == Create model configuration ==
    model_config = ModelConfig(
        numeric_impl=prsrargs.nume_impl,
        IS_RESIDUAL=prsrargs.tempo_residuals,
        Omega_ref=omgc_0_par,
        Omega_dot_ref=omgc_dot_par,
    )
    analyse_twocomp(
        times,
        phases,
        R_phase,
        omgc_0_par,
        omgc_dot_par,
        src_config,
        model_config,
        prsrargs.out_directory,
        prsrargs.tag,
        tempo_psr=tempo_psr,
    )

    # analyse_onecompWTN(
    #     times,
    #     phases,
    #     R_phase,
    #     omgc_0_par,
    #     omgc_dot_par,
    #     src_config,
    #     model_config,
    #     prsrargs.out_directory,
    #     f"{prsrargs.tag}_WTNnullHypothesis",
    #     tempo_psr=tempo_psr,
    # )


if __name__ == "__main__":
    main()
