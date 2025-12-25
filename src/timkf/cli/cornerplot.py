import os
import argparse
import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bilby.core.prior import Prior, LogUniform
from bilby.core.result import read_in_result, plot_multiple, Result
from ..misc import update_corner_plots, tau_eff_squared
from .utils import load_toml, apply_cli_overrides


def update_indiv_corner_plots():
    config = configparser.ConfigParser()
    config.read("analysis_config.ini")

    outdir_parnt = config["rerun_using_existing_data"]["outdir_parnt"]
    outdir_child = config["rerun_using_existing_data"]["outdir_child"]
    run_slurm_file = config["rerun_using_existing_data"]["run_slurm_file"]

    outdir = os.path.join(outdir_parnt, outdir_child)

    mcomp_csvs = [
        f for f in os.listdir(outdir) if f.endswith(".csv") and "model_comparison" in f
    ]
    redish_psrs = []
    for csv in mcomp_csvs:
        df = pd.read_csv(os.path.join(outdir, csv), index_col="PSRJ")
        redish_psrs.append(
            list(df[df["logBF_2Cmpnt_vs_WTNnullHypothesis"] > 0.0].index)
        )

    redish_psrs = set([item for sublist in redish_psrs for item in sublist])

    update_corner_plots(
        datadir_parnt=outdir,
        shell_run_python_file=run_slurm_file,
        only_psrs=redish_psrs,
    )


def get_axes_label(result: Result, key: str) -> str:
    """
    Get the label for the axes based on the prior type.
    """
    DEFAULT_KEY_UNITS = {
        "ratio": None,
        "tau": "s",
        "Qc": r"rad$^2$\,s$^{-3}$",
        "Qs": r"rad$^2$\,s$^{-3}$",
        "omgc_0": r"rad\,s$^{-1}$",
        "omgc_dot": r"rad\,s$^{-2}$",
        "sigmaratio": None,
        "tau_eff": "s",
    }
    prior_obj: Prior = result.priors[key]
    ax_label = prior_obj.latex_label
    unit = prior_obj.unit or DEFAULT_KEY_UNITS.get(key, None)
    if unit is not None:
        if unit == "s":
            ax_label += r"$\,\rm{s}^{-1}$"
        else:
            ax_label += f"/{unit}"
    if isinstance(prior_obj, LogUniform):
        ax_label = f"log$_{{10}}$({ax_label})"
    return ax_label


def combine_indiv_corner_plots(
    psrs: list,
    parameters: list[str] = ["ratio", "tau", "Qc", "Qs", "sigmaratio", "tau_eff"],
    main_parameters: list[str] = ["tau", "Qc", "Qs"],
    data_dir: str = None,
):
    if data_dir is None:
        raise ValueError(
            "data_dir must be provided to combine_indiv_corner_plots. "
            "This is the directory where the results are stored."
        )

    irrelavant_dirs = ["hrchy", "outdir_log", "outdir_err"]
    result_dirs = [
        d
        for d in os.listdir(data_dir)
        if d not in irrelavant_dirs and d.split("_")[1] in psrs
    ]
    print(f"{result_dirs=}")
    # colors = plt.get_cmap("seismic", len(result_dirs))
    results = []
    axes_labels = {}
    for d in result_dirs:
        json_file = os.path.join(data_dir, d, f"{d.split('_')[1]}_result.json")
        print(f"Reading in {json_file}")
        result = read_in_result(json_file)
        for key in result.search_parameter_keys:
            # convert to log scale
            prior_obj = result.priors[key]
            if isinstance(prior_obj, LogUniform):
                result.posterior[key] = np.log10(result.posterior[key])
            axes_labels[key] = get_axes_label(result, key)
        result.posterior["sigmaratio"] = (
            result.posterior["Qs"]
            - result.posterior["Qc"]
            + 2 * result.posterior["ratio"]
        ) / 2
        axes_labels["sigmaratio"] = "log$_{10}$($\sigma_{\\rm s} / \sigma_{\\rm c}$)"
        result.posterior["tau_eff"] = (
            np.log10(
                tau_eff_squared(
                    10 ** result.posterior["ratio"],
                    10 ** result.posterior["Qs"],
                    10 ** result.posterior["Qc"],
                    10 ** result.posterior["tau"],
                )
            )
            / 2
        )
        axes_labels["tau_eff"] = "log$_{10}$($\\tau_{\\rm eff}\,\\rm{s}^{-1}$)"

        results.append(result)

    plot_suffix = "_combined_cornerplots" if len(psrs) > 1 else "_cornerplot"
    plot_suffix += (
        "_" + "-".join(parameters) if set(parameters) != set(main_parameters) else ""
    )
    plot_name = os.path.join("./docs/paperplots/", f"{'-'.join(psrs)}{plot_suffix}.png")
    if len(psrs) == 1:
        from matplotlib import rcParams

        result: Result = results[0]
        result.plot_corner(
            filename=plot_name,
            parameters=parameters,
            labels=[axes_labels[key] for key in parameters],
            titles=None,
            # show_titles=True,
            # title_fmt=".1e",
            # quantiles=[0.16, 0.5, 0.84],
            title_kwargs={"fontsize": rcParams["axes.titlesize"]},
        )
    else:
        plot_multiple(
            results,
            filename=plot_name,
            # colours=[colors(i) for i in range(len(result_dirs))],
            titles=None,
            labels=["PSR " + d.split("_")[1].replace("-", "$-$") for d in result_dirs],
            parameters=parameters,
            corner_labels=[axes_labels[key] for key in parameters],
            colours=["tab:blue", "tab:orange"] if len(psrs) == 2 else None,
        )


def combine_hrchy_corner_plots(
    plot_models: list[str] = [
        "PowerLawCorrAgainstOmgcOmgcdotHyperModel-101psrs_dynesty_exclude_msp_exclude_mag"
    ],
    outdir_hrchy: str = None,
):
    if outdir_hrchy is None:
        raise ValueError(
            "outdir_hrchy must be provided to combine_hrchy_corner_plots. "
            "This is the directory where the hierarchical results are stored."
        )

    result_dirs = [
        d
        for d in os.listdir(outdir_hrchy)
        if not d.startswith("legacy") and d.startswith("PowerLaw")
    ]
    result_dirs.sort()
    if plot_models:
        result_dirs = [
            d for d in result_dirs if any(model in d for model in plot_models)
        ]
    # colors = plt.get_cmap("seismic", len(result_dirs))
    results = []
    for i, d in enumerate(result_dirs):
        json_file = os.path.join(outdir_hrchy, d, f"{d}_result.json")
        print(f"Reading in {json_file}")
        result = read_in_result(json_file)
        results.append(result)

    if len(results) == 1:
        result: Result = results[0]
        result.plot_corner(
            filename=os.path.join(
                "./docs/paperplots/", f"{d.split('/')[-1]}_corner.png"
            )
        )
        return
    plot_multiple(
        results,
        filename=os.path.join("./docs/paperplots/", "hrchy_combined_cornerplots.png"),
        # colours=[colors(i) for i in range(len(result_dirs))],
        titles=None,
        labels=[d.split("-")[1] for d in result_dirs],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Update corner plots for the UTMOST pulsar population analysis."
    )
    parser.add_argument(
        "--cfg_path",
        "-c",
        type=str,
        default="configs/cornerplot.toml",
        help="Path to the configuration file for the corner plot.",
    )
    parser.add_argument(
        "--style",
        "-s",
        choices=[None, "science", "ieee"],
        default="science",
        help="Style of plots to use with SciencePlots.",
    )
    parser.add_argument(
        "--function",
        "-f",
        choices=["indiv", "hrchy"],
        required=True,
        help="Which function to run: 'indiv' for individual-level corner plots, 'hrchy' for corner plots of hierarchical parameters.",
    )
    parser.add_argument(
        "--objects",
        "-o",
        nargs="+",
        help="List of objects to plot together. Only used if --function is 'indiv'. If not provided, a set of pre-selected objects will be used. If all objects are to be plotted, use 'all'.",
    )
    parser.add_argument(
        "--params",
        "-p",
        nargs="+",
        default=["tau", "Qc", "Qs"],
        help="List of parameters to include in the corner plot. Default is ['tau', 'Qc', 'Qs'].",
    )
    parser.add_argument(
        "--override",
        "-r",
        action="append",
        default=[],
        help="Override configuration parameters in the form 'a.b.c=d'. Can be used multiple times to override multiple parameters.",
    )
    args = parser.parse_args()

    config = load_toml(args.cfg_path)
    config = apply_cli_overrides(config, args.override)
    dataset_cfg = config["dataset"]

    if args.style:
        import scienceplots  # noqa

        plt.style.use(args.style)

    if args.function == "indiv":
        # Update rcParams if specified in the config
        if my_rcparams := config.get("indiv", {}).get("rcparams"):
            plt.rcParams.update({"font.size": my_rcparams.get("font_size")})
        if not args.objects:
            # Default selection of pulsars
            args.objects = config["indiv"]["selected_objects"]
        elif args.objects == ["all"]:
            # TODO: Implement logic to fetch all pulsars
            raise NotImplementedError("Fetching all pulsars is not implemented yet.")
        else:
            # Make args.objects a list of lists
            print(f"Using provided objects: {args.objects}")
            args.objects = [args.objects]

        for psrs in args.objects:
            combine_indiv_corner_plots(
                psrs=psrs,
                parameters=args.params,
                data_dir=dataset_cfg["data_dir"],
            )
    # combine_indiv_corner_plots(["J1141-6545"], parameters=["tau", "ratio", "omgc_0"])
    elif args.function == "hrchy":
        # Update rcParams if specified in the config
        if my_rcparams := config.get("hrchy", {}).get("rcparams"):
            plt.rcParams.update({"font.size": my_rcparams.get("font_size")})
        outdir_hrchy = os.path.join(
            dataset_cfg["data_dir"], dataset_cfg["hrchy_subdir"]
        )
        combine_hrchy_corner_plots(outdir_hrchy=outdir_hrchy)
    else:
        raise ValueError(f"Unknown function: {args.function}")

    # update_indiv_corner_plots()


if __name__ == "__main__":
    main()
