import os
import argparse
import configparser
from tqdm import tqdm
import bilby
import numpy as np
import pandas as pd
from pandas import DataFrame
import scipy.stats as sp_stats
import re  # for caption replacement in latex table
import libstempo as T2
import itertools as it
from typing import Union, Literal
from timkf.misc import (
    find_sorted_subdirs,
    B_surf,
    is_recycled_millisecond,
    is_magnetar,
    timing_noise_strength,
    tau_eff_squared,
)
from timkf.constants import DAY_TO_SECONDS
from timkf.hyper import HYPER_MODEL_MAPPING_DICT

cwd = "./"


def get_model_comparison_setup_from_config(config: configparser.ConfigParser):
    parnt_dirs_null = []
    child_dirs_null = []
    parnt_dirs = []
    child_dirs = []
    if_toas_ = []
    save_csvs = []
    for key in config["model_comparison"]:
        if key.startswith("outdir_parnt_null"):
            parnt_dirs_null.append(config["model_comparison"][key])
        elif key.startswith("outdir_child_null"):
            child_dirs_null.append(config["model_comparison"][key])
        elif key.startswith("outdir_parnt"):
            parnt_dirs.append(config["model_comparison"][key])
        elif key.startswith("outdir_child"):
            child_dirs.append(config["model_comparison"][key])
        elif key.startswith("if_toas"):
            if_toas_.append(config["model_comparison"].getboolean(key))
        elif key.startswith("save_csv"):
            save_csvs.append(config["model_comparison"].getboolean(key))

    if len(parnt_dirs) == 1:
        print(
            "Only ONE parent directory is given, broadcasting to all child directories"
        )
        parnt_dirs = parnt_dirs * len(child_dirs)
    if len(save_csvs) == 1:
        print("Only ONE save_csv is given, broadcasting to all child directories")
        save_csvs = save_csvs * len(child_dirs)
    if len(if_toas_) == 1:
        print("Only ONE if_toas is given, broadcasting to all child directories")
        if_toas_ = if_toas_ * len(child_dirs)

    assert len(parnt_dirs_null) == 1 and len(child_dirs_null) == 1, (
        "Only ONE null hypothesis directory is allowed"
    )
    return (
        parnt_dirs_null[0],
        child_dirs_null[0],
        parnt_dirs,
        child_dirs,
        if_toas_,
        save_csvs,
    )


def bayes_factors(df: pd.DataFrame):
    columns = [col for col in df.columns if col.startswith("log_evidence_")]
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            bayes_factor_col = f"logBF_{col2.split('_')[2]}_vs_{col1.split('_')[2]}"
            df[bayes_factor_col] = df[col2] - df[col1]
            if "NTOAs" in df.columns:
                BF_factor_col_per_toa = (
                    f"logBF_per_TOA_{col2.split('_')[2]}_vs_{col1.split('_')[2]}"
                )
                df[BF_factor_col_per_toa] = df[bayes_factor_col] / df["NTOAs"]
    return df


def create_model_data(
    data_dir, ref_dir, save_csv=True, if_toas=False, csv_name="model_comparison.csv"
):
    parfile_parnt_dir = os.path.join(cwd, "data/TimingDataRelease1/pulsars")
    subdirs = find_sorted_subdirs(data_dir)
    psr_names = []
    psr_data = []

    for sd in tqdm(subdirs, desc="Processing JSON files of each PSR"):
        # get result.json files from both data and ref directories
        _, psr_name = sd.split("_")
        psr_names.append(psr_name)
        subdir_path_data = os.path.join(data_dir, sd)
        subdir_path_ref = os.path.join(ref_dir, sd)
        json_file_data = [
            f for f in os.listdir(subdir_path_data) if f.endswith("result.json")
        ]
        json_file_ref = [
            f for f in os.listdir(subdir_path_ref) if f.endswith("result.json")
        ]
        assert len(json_file_data) == 1 and len(json_file_ref) == 1, (
            f"More than one JSON file found in {subdir_path_data} or {subdir_path_ref}"
        )
        json_files_dict = dict(ref=json_file_ref[0], data=json_file_data[0])

        model_dataDict = {}
        # will run out of memory if use T2.tempopulsar
        if if_toas:
            with open(
                os.path.join(parfile_parnt_dir, psr_name, f"{psr_name}.par")
            ) as f:
                for line in f:
                    if line.startswith("NTOA"):
                        model_dataDict["NTOAs"] = int(line.split()[1])
                    elif line.startswith("START"):
                        tstart = float(line.split()[1])
                    elif line.startswith("FINISH"):
                        tfinish = float(line.split()[1])
                model_dataDict["TSTART"] = tstart
                model_dataDict["TFINISH"] = tfinish
                model_dataDict["Tobs"] = tfinish - tstart
        for k, jf in json_files_dict.items():
            # e.g. J0152-1637_WTNnullHypothesis_result.json
            label = "_".join(jf.split("_")[1:-1]) if k == "ref" else "2Cmpnt"
            subdir_path = subdir_path_ref if k == "ref" else subdir_path_data

            result = bilby.read_in_result(os.path.join(subdir_path, jf))
            model_dataDict[f"log_evidence_{label}"] = result.log_evidence
            if k == "data":
                check_cols = ["tau", "Qc", "Qs", "tauprime", "ratio"]
                posteriors: pd.DataFrame = result.posterior
                quantiles = posteriors.quantile([0.5, 0.16, 0.84])
                for col in check_cols:
                    if col not in posteriors.columns:
                        continue
                    model_dataDict[f"logq_{col}"] = np.log10(
                        quantiles[col].values
                    ).tolist()
                    model_dataDict[f"log_{col}-dist"] = category_kde(
                        np.log10(posteriors[col]),
                        np.log10(result.priors[col].minimum),
                        np.log10(result.priors[col].maximum),
                    )
                if "logq_taueff" not in model_dataDict:
                    model_dataDict["logq_taueff"] = np.quantile(
                        np.log10(
                            tau_eff_squared(
                                posteriors["ratio"],
                                posteriors["Qc"],
                                posteriors["Qs"],
                                posteriors["tau"],
                            )
                        )
                        / 2,
                        [0.5, 0.16, 0.84],
                    ).tolist()

                log_TN_strength = timing_noise_strength(
                    posteriors["omgc_0"],
                    posteriors["Qc"],
                    posteriors["Qs"],
                    posteriors["tau"],
                    posteriors["ratio"],
                    t_obs=model_dataDict["Tobs"] * DAY_TO_SECONDS,
                    n_obs=model_dataDict["NTOAs"],
                    return_log10=True,
                )
                model_dataDict["logq_TN"] = np.quantile(
                    log_TN_strength, [0.5, 0.16, 0.84]
                ).tolist()
        model_dataDict["omgc_0"] = result.posterior["omgc_0"].median()
        model_dataDict["omgc_dot"] = result.posterior["omgc_dot"].median()
        psr_data.append(model_dataDict)

    df = pd.DataFrame(psr_data, index=psr_names)
    df.sort_index(axis="columns", ascending=False, inplace=True)
    df.index.name = "PSRJ"
    bayes_factors(df)
    if save_csv:
        df.to_csv(os.path.join(data_dir, csv_name), index=True)
    return df


def get_utmost_bayes_factor_table(
    file_path="data/TimingDataRelease1/utmost_supplementary_files/rednoise_table_B1.csv",
):
    file_path = os.path.join(cwd, file_path)
    df = pd.read_csv(file_path, delimiter=";", skiprows=[1])
    df.columns = df.columns.str.replace("# ", "")

    return df


def get_utmost_ephemerides_table(
    file_path="data/TimingDataRelease1/utmost_supplementary_files/ephemerides_table_A1.csv",
):
    file_path = os.path.join(cwd, file_path)
    df = pd.read_csv(file_path, delimiter=";", skiprows=[1])
    df.columns = df.columns.str.replace("# ", "")

    return df


def category_kde(data, prior_logmin=-np.inf, prior_logmax=np.inf):
    kde = sp_stats.gaussian_kde(data)
    x = np.linspace(prior_logmin, prior_logmax, 1000, endpoint=True)
    kde_values = kde(x)

    half_max = np.max(kde_values) / 2
    idx_fwhm = np.where(np.diff(np.sign(kde_values - half_max)))[0]

    # if no FWHM found, meaning the distribution is roughly flat -> not peaky
    if len(idx_fwhm) == 0:
        return "-"
    # if value at left boundary is above half max, consider it as not peaky
    if kde_values[0] > half_max:
        return "-"
    if np.isclose(x[np.min(idx_fwhm)], prior_logmin, atol=5e-1):
        # if the left boundary of fwhm is close to the prior minimum, consider it as not peaky
        return "-"
    # if value at right boundary is above half max, consider it as rr
    if kde_values[-1] > half_max:
        return "rr"
    if np.isclose(x[np.max(idx_fwhm)], prior_logmax, atol=5e-1):
        # if the right boundary of fwhm is close to the prior maximum, consider it as not peaky and mark it as rr
        return "rr"
    return "pky"


def filter_nonwhite(
    df_utmost: pd.DataFrame = None,
    df_two_cmpnt: pd.DataFrame = None,
    csv_name=None,
    return_all=False,
):
    print(f"{'=' * 20} Filter non-white noise candidates {'=' * 20}")
    if df_utmost is None:
        df_utmost = get_utmost_bayes_factor_table()
        df_utmost.set_index("PSRJ", inplace=True)
        df_utmost_ephemerides = get_utmost_ephemerides_table()
        df_utmost_ephemerides.set_index("PSRJ", inplace=True)
    if df_two_cmpnt is None:
        assert csv_name is not None, "csv_name is not provided"
        df_two_cmpnt = pd.read_csv(csv_name, index_col="PSRJ")
        print(f"Loaded {csv_name} with {len(df_two_cmpnt)} pulsars")
    # == filter out the glitching pulsars ==
    glitch_psrs = df_utmost.index.difference(df_two_cmpnt.index)
    utmost_nonglitch = df_utmost.drop(glitch_psrs)
    print(f"Number of UTMOST nonglitching pulsars: {len(utmost_nonglitch)}\n")

    NONWHITE_THRES = 5.0
    print(f"{NONWHITE_THRES = }")
    utmost_nonwhite = df_utmost[df_utmost["Model"] != "WTN"]
    utmost_PLRNstrong = utmost_nonwhite[utmost_nonwhite["lnBF"] > NONWHITE_THRES]
    print(
        f"Number of UTMOST pulsars that strongly favour PLRN out of {len(df_utmost)} objects: {len(utmost_PLRNstrong)}\n"
    )
    F0 = df_utmost_ephemerides["F0"]
    F1 = df_utmost_ephemerides["F1"] * 1e-15
    df_utmost_recycled = df_utmost_ephemerides[is_recycled_millisecond(F0, F1)]
    print(
        f"Number of UTMOST recycled pulsars: {len(df_utmost_recycled)}:\n{df_utmost_recycled.index}\nNumer of UTMOST recycled pulsars that strongly favour PLRN: {len(utmost_PLRNstrong[utmost_PLRNstrong.index.isin(df_utmost_recycled.index)])}\n"
    )
    df_utmost_magnetar = df_utmost_ephemerides[
        is_magnetar(B_surf(F0 * 2 * np.pi, F1 * 2 * np.pi))
    ]
    print(f"Number of UTMOST magnetars: {len(df_utmost_magnetar)}\n")
    utmost_nonglitch_nonwhitestrong = utmost_nonglitch[
        utmost_nonglitch["lnBF"] > NONWHITE_THRES
    ]
    print(
        f"Number of UTMOST non-glitching pulsars that strongly favour PLRN: {len(utmost_nonglitch_nonwhitestrong)}\n"
    )
    # == filter two-component candidates ==
    two_cmpnt_nonwhite = df_two_cmpnt[
        df_two_cmpnt["logBF_2Cmpnt_vs_WTNnullHypothesis"] > NONWHITE_THRES
    ]
    print(f"Number of two-component candidates (strong): {len(two_cmpnt_nonwhite)}")
    # == intersection and difference ==
    intersection = two_cmpnt_nonwhite.index.intersection(
        utmost_nonglitch_nonwhitestrong.index
    )
    print(
        f"Intersection of two-component candidates (strong) and UTMOST non-glitching PLRN pulsars (strong):{len(intersection)}\n{intersection}"
    )
    print(
        f"In two-component candidates (strong) but not in UTMOST non-glitching PLRN pulsars (strong): \n{two_cmpnt_nonwhite.index.difference(utmost_nonglitch_nonwhitestrong.index)}"
    )
    print(
        f"In UTMOST non-glitching PLRN pulsars (strong) but not in two-component candidates (strong) (glitched psrs removed): \n{utmost_nonglitch_nonwhitestrong.index.difference(two_cmpnt_nonwhite.index)}"
    )
    print("Non-uniform distributed tau posteriros for non-white noise candidates:")
    print(two_cmpnt_nonwhite["log_tau-dist"].value_counts())
    print("Non-uniform distributed tau posteriros for all candidates:")
    print(df_two_cmpnt["log_tau-dist"].value_counts())
    if return_all:
        return (
            two_cmpnt_nonwhite,
            utmost_nonwhite,
            df_two_cmpnt,
            utmost_nonglitch,
            utmost_nonglitch_nonwhitestrong,
        )
    return two_cmpnt_nonwhite, utmost_nonwhite


def prepare_df_2cmpnt(df_2cmpnt: pd.DataFrame, params: list, params_latex: dict):
    IRRELEVANT_COLS = [
        "TSTART",
        "TFINISH",
        "Tobs",
        "log_evidence_WTNnullHypothesis",
        "log_evidence_2Cmpnt",
        "NTOAs",
        "logBF_per_TOA_2Cmpnt_vs_WTNnullHypothesis",
        "logq_ratio",
        "log_ratio-dist",
        "logq_taueff",
    ]
    df_2cmpnt = df_2cmpnt.drop(columns=IRRELEVANT_COLS)
    df_2cmpnt["Model"] = df_2cmpnt["logBF_2Cmpnt_vs_WTNnullHypothesis"].apply(
        lambda x: "2C" if x > 5 else "WTN"
    )
    df_2cmpnt.insert(0, "Model", df_2cmpnt.pop("Model"))
    # clean up the columns for df_2cmpnt
    for p in params:
        # convert string to list of floats
        df_2cmpnt[f"logq_{p}"] = (
            df_2cmpnt[f"logq_{p}"]
            .str.strip("[]")
            .str.split(", ")
            .apply(lambda x: [float(i) for i in x])
        )
        # format the presentation of the log quantiles
        median_with_error = [
            "$%.1f^{+%.1f}_{-%.1f}$" % (m, h - m, m - low)
            for m, low, h in df_2cmpnt[f"logq_{p}"]
        ]

        # flag if lognorm distribution is preferred against loguniform distribution
        def pky_flag(flag):
            if flag == "pky":
                return "Y"
            elif flag == "rr":
                return "RR"
            else:
                return "N"

        df_2cmpnt[f"log$_{{10}}$({params_latex[p]})"] = median_with_error
        if p == "tau":
            df_2cmpnt[f"dist{p}-peaky?"] = df_2cmpnt[f"log_{p}-dist"].apply(pky_flag)
    psr_logtau_pky = df_2cmpnt[df_2cmpnt["log_tau-dist"].str.contains("Y")].index
    # drop the columns that are not needed
    df_2cmpnt = df_2cmpnt.drop(
        columns=[f"log_{p}-dist" for p in params] + [f"logq_{p}" for p in params]
    )
    if "logq_TN" in df_2cmpnt.columns:
        df_2cmpnt["logq_TN"] = (
            df_2cmpnt["logq_TN"]
            .str.strip("[]")
            .str.split(", ")
            .apply(lambda x: [float(i) for i in x])
        )
        # format the presentation of the log quantiles
        median_with_error = [
            "$%.1f^{+%.1f}_{-%.1f}$" % (m, h - m, m - low)
            for m, low, h in df_2cmpnt["logq_TN"]
        ]
        df_2cmpnt["log$_{10} \sigma_{\\rm TN}^2$"] = median_with_error
        # drop the columns that are not needed
        df_2cmpnt = df_2cmpnt.drop(columns=["logq_TN"])
    # rename the columns
    df_2cmpnt.columns = df_2cmpnt.columns.str.replace(
        "logBF_2Cmpnt_vs_WTNnullHypothesis", "$\ln\mathfrak{B}_{\\rm BF}$"
    )

    return df_2cmpnt, psr_logtau_pky


def create_latex_table_indiv(
    df_2cmpnt: pd.DataFrame,
    df_utmost: pd.DataFrame,
    params: list = ["tau", "Qc", "Qs"],
    params_latex: dict = {"tau": "$\\tau$", "Qc": "$Q_{\\rm c}$", "Qs": "$Q_{\\rm s}$"},
    asterisk_objects: list[str] = None,
    dagger_objects: list[str] = None,
    glitched_objects: list[str] = None,
    remove_objects: list[str] = None,
):
    # prepare the df_2cmpnt
    df_2cmpnt, psr_logtau_pky = prepare_df_2cmpnt(df_2cmpnt, params, params_latex)

    # clean up the columns for df_utmost
    df_utmost = df_utmost[["Model", "lnBF"]]
    df_utmost.columns = ["Model", "$\ln \mathfrak{B}_{\\rm BF}$"]

    # join = 'outer' to keep all the non-overlapping psrs in df_2cmpnt
    df_combined = pd.concat([df_2cmpnt, df_utmost], axis=1, join="outer")

    # modify the index to match the asterisk and dagger objects
    if asterisk_objects is None:
        asterisk_objects = []
    if dagger_objects is None:
        dagger_objects = []
    if glitched_objects is None:
        glitched_objects = []
    if remove_objects is not None:
        df_removed = df_combined.loc[remove_objects]
        df_combined = df_combined.drop(index=remove_objects, errors="ignore")

    def mapper_func(x):
        mark = ""
        if x in glitched_objects:
            mark += r"$^{\rm g}$"
        if x in asterisk_objects:
            mark += "$^*$"
            # return f"{x}$^*$"
        elif x in dagger_objects:
            mark += "$^\dagger$"
            # return f"{x}$^\dagger$"
        return f"{x}{mark}"

    df_combined.index = df_combined.index.map(mapper_func)
    df_combined.index = df_combined.index.map(lambda x: x.replace("-", "$-$"))

    # set the second column header for hierarchical display
    second_header = [""] * (
        len(df_combined.drop(columns=["omgc_0", "omgc_dot"]).columns)
        - len(df_utmost.columns)
    ) + ["\citet{LowerEtAl2020}"] * len(df_utmost.columns)

    _make_canonical_recycled_magetar_table(
        df_combined, "msp", second_header=second_header, latex_label=False
    )
    _make_canonical_recycled_magetar_table(
        df_combined, "magnetar", second_header=second_header, latex_label=False
    )
    secondary_caption = "continued from previous page."
    _make_canonical_recycled_magetar_table(
        df_combined,
        "canonical",
        second_header=second_header,
        latex_label=False,
        main_caption=None,
        secondary_caption=secondary_caption,
        longtable=True,
    )
    # _make_canonical_recycled_magetar_table(
    #     df_removed, "glitching", second_header=second_header, latex_label=False
    # )


def _make_canonical_recycled_magetar_table(
    data: DataFrame,
    psr_type: Union[Literal["msp", "magnetar", "canonical"]],
    table_name=None,
    latex_label: bool | None = None,
    second_header=None,
    main_caption=None,
    secondary_caption="continued from previous page",
    longtable: bool = False,
) -> tuple[Literal["latex_code"], DataFrame]:
    VALID_PSR_TYPES = ["msp", "magnetar", "canonical"]
    if psr_type not in VALID_PSR_TYPES:
        raise ValueError(
            f"psr_type must be one of {VALID_PSR_TYPES}, got {psr_type} instead."
        )

    _is_recycled = is_recycled_millisecond(
        data["omgc_0"] / (2 * np.pi), data["omgc_dot"] / (2 * np.pi)
    )
    _is_magnetar = is_magnetar(B_surf(data["omgc_0"], data["omgc_dot"]))
    if psr_type == "msp":
        data = data[_is_recycled]
    elif psr_type == "magnetar":
        data = data[_is_magnetar]
    # canonical pulsars are those that are not recycled or magnetars
    elif psr_type == "canonical":
        data = data[~_is_recycled & ~_is_magnetar]
    # drop the columns that are not needed for display
    data = data.drop(columns=["omgc_0", "omgc_dot"])
    # set the second column header for hierarchical display
    if second_header is not None:
        data.columns = pd.MultiIndex.from_arrays([second_header, data.columns])

    # set the table name
    if table_name is None:
        table_name = f"model_comparison_{psr_type}"
    # default the latex_label if not provided, unless set to False
    if latex_label is None or latex_label is True:
        latex_label = f"tab:{table_name}"
    elif latex_label is False:
        latex_label = None

    LONGTABLE = longtable
    latex_code = data.to_latex(
        label=latex_label,
        column_format="l" + "c" * (len(data.columns) - 2) + "rr"
        if not LONGTABLE
        else None,
        float_format="%.1f",
        na_rep="-",
        multicolumn=True,
        multicolumn_format="c",
        caption=secondary_caption if LONGTABLE else main_caption,
        longtable=LONGTABLE,
    )
    if LONGTABLE:
        replace_str = r"\\caption{%s}" % main_caption
        if main_caption is None:
            replace_str = ""
        # replace the first occurance of caption with main caption
        latex_code = re.sub(
            r"\\caption\{.*?\}",
            replace_str,
            latex_code,
            count=0,
        )
    # extra formatting (make sure not wrapped in table environment)
    latex_code = re.sub(
        r"dist[^-]*-peaky\?",
        r"Peaky?",
        latex_code,
    )
    add_units = ["[s]", "[rad$^2$s$^{-3}]$", "[rad$^2$s$^{-3}]$", "[s$^2]$"]
    unit_row = "PSR J"
    for _, col in data.columns:
        if col.startswith("log$_{10}"):
            unit_row += f" & {add_units.pop(0)}"
        else:
            unit_row += " &"
    unit_row = unit_row.removesuffix(" &")  # remove the last &
    latex_code = latex_code.replace(
        f"PSRJ & {' & '.join([''] * (len(data.columns) - 1))}",
        unit_row,
    )
    latex_code = re.sub(
        r"\\begin\{.*?table\*?\}\{(.*?)\}",
        "",  # remove the table environment
        latex_code,
    )
    latex_code = re.sub(
        r"\\end\{.*?table\*?\}",
        "",  # remove the table environment
        latex_code,
    )
    latex_code = re.sub(
        r"J(\d{4})",
        r"\1",  # replace Jxxxx with xxxx
        latex_code,
    )
    # replace \rule with \hline
    latex_code = re.sub(
        r"\\(top|mid|bottom)rule",
        r"\\hline",
        latex_code,
    )

    with open(f"./docs/papertabs/{table_name}.tex", "w") as f:
        f.write(latex_code)

    return latex_code, data


def _prepare_df_hyper(result_path: str, incl_BF: bool = True):
    # read in the result
    result: bilby.result.Result = bilby.result.read_in_result(result_path)
    # get the posterior
    posterior: pd.DataFrame = result.posterior

    posterior = posterior.drop(columns=["log_likelihood", "log_prior"])
    pquant_df = posterior.quantile([0.5, 0.16, 0.84])

    hm_info_df = pd.DataFrame(columns=["NewModelName"])
    if incl_BF:
        hm_info_df = pd.DataFrame(
            {"NewModelName": result.log_bayes_factor}, index=["lnBF"]
        )
    for col in posterior.columns:
        mid, low, high = pquant_df[col].values
        hm_info_df.loc[col] = f"${mid:.2f}^{{+{high - mid:.2f}}}_{{-{mid - low:.2f}}}$"
    return hm_info_df, result


def _assign_hypermodel_names(hm_dir: str, model_num: int, hm_names_dict: dict):
    if hm_dir.startswith("Log10norm"):
        hm_index = "1"
    if hm_dir.startswith("Log10normLog10unifFixedOneBdryPreCategorised"):
        hm_index = "3"
    elif hm_dir.startswith("Log10normLog10unifFixedOneBdry"):
        hm_index = "2ii"
    elif hm_dir.startswith("TwoLog10norm"):
        hm_index = "2i"
    else:
        hm_index = f"{model_num + 1}"
    hm_names_dict[hm_index] = hm_dir.split("_")[0].replace("HyperModel", "")

    return hm_index, hm_names_dict


def create_latex_table_hyper(
    hrchy_parnt_dir,
    params: list[str] = ["tau", "Qc", "Qs"],
    pars_latex: dict[str, str] = {
        "tau": "$\\tau$",
        "Qc": "$Q_{\\rm c}$",
        "Qs": "$Q_{\\rm s}$",
    },
    hp_pars_latex: dict[str, str] = {
        "C2": "$C_{2}$",
        "a": "$a$",
        "b": "$b$",
        "sigma": "$\\sigma$",
    },
    only_hpmodels: list[str] = None,
    ignore_hpmodels: list[str] = None,
    combine: bool = True,
):
    if ignore_hpmodels is None:
        ignore_hpmodels = []
    ignore_hpmodels = [HYPER_MODEL_MAPPING_DICT[hm].__name__ for hm in ignore_hpmodels]
    ignore_hpmodels.append("legacy_hypermodel_inference")
    if only_hpmodels is None:
        only_hpmodels = []
    only_hpmodels = [HYPER_MODEL_MAPPING_DICT[hm].__name__ for hm in only_hpmodels]

    # always expect joint parameter estimation
    param_comb = "-".join(params)
    target_path = os.path.join(hrchy_parnt_dir, "hrchy", param_comb)
    hpmodel_dirs = find_sorted_subdirs(target_path)
    # filter out ignored hpmodels
    hpmodel_dirs = [
        hm for hm in hpmodel_dirs if not any(ig_hm in hm for ig_hm in ignore_hpmodels)
    ]
    # keep only the hpmodels specified in only_hpmodels
    if only_hpmodels:
        hpmodel_dirs = [
            hm for hm in hpmodel_dirs if any(only_hm in hm for only_hm in only_hpmodels)
        ]

    hpmodel_dirs.sort()
    hm_names_dict = {}
    table_df = pd.DataFrame()
    for model_num, hm_dir in enumerate(hpmodel_dirs):
        result_path = os.path.join(target_path, hm_dir, f"{hm_dir}_result.json")
        if not os.path.exists(result_path):
            continue
        hm_info_df, result = _prepare_df_hyper(result_path, incl_BF=False)
        hm_index, hm_names_dict = _assign_hypermodel_names(
            hm_dir, model_num, hm_names_dict
        )
        # change the default new column name
        psr_num_match = re.search(r"(\d+)psr", hm_names_dict[hm_index])
        hm_info_df.rename(
            columns={
                "NewModelName": f"HM-{psr_num_match.group()}"
                if psr_num_match
                else f"HM {hm_index}",
            },
            inplace=True,
        )
        if combine:
            table_df = pd.concat([table_df, hm_info_df], axis=1)
        else:
            hp_pars = sorted(
                list(
                    set(
                        [
                            idx.replace(f"_{par}", "")
                            for par in params
                            for idx in hm_info_df.index
                            if par in idx
                        ]
                    )
                )
            )
            table_df = pd.DataFrame(index=hp_pars, columns=params)
            for par, hp_par in it.product(params, hp_pars):
                orig_idx = f"{hp_par}_{par}"
                if orig_idx in hm_info_df.index:
                    table_df.loc[hp_par, par] = hm_info_df.loc[orig_idx].values

            table_df = pd.DataFrame(table_df)
            table_df.rename(columns=pars_latex, inplace=True)
            table_df.rename(index=hp_pars_latex, inplace=True)
            # set the environment to table* in case for two-column format
            pd.options.styler.latex.environment = "table"
            # save to latex
            table_latex_str = table_df.to_latex(
                float_format="%.2f",
                label=f"tab:hypermodel_comparison_{param_comb}_{psr_num_match.group()}",
                na_rep="-",
            )
            table_latex_str = table_latex_str.replace(r"\begin{table}", "").replace(
                r"\end{table}", ""
            )
            with open(
                f"./docs/papertabs/hypermodel_comparison_{param_comb}_{psr_num_match.group()}.tex",
                "w",
            ) as f:
                f.write(table_latex_str)

    if combine:
        # sort the table by columns
        table_df = table_df.reindex(sorted(table_df.columns), axis=1)
        # format the index
        table_df.index = result.get_latex_labels_from_parameter_keys(table_df.index)
        # set the environment to table* in case for two-column format
        pd.options.styler.latex.environment = "table"
        # save to latex
        table_df.to_latex(
            f"./docs/papertabs/hypermodel_comparison_{param_comb}.tex",
            float_format="%.2f",
            caption=f"Table of hypermodels for {', '.join(list(pars_latex.values())[:-1])}, and {list(pars_latex.values())[-1]} with log Bayes factor and the median values and errors specified by the 95\% confidence interval for hyperparameters. {', '.join(list(table_df.columns)[:-1])} and {table_df.columns[-1]} are the hypermodels applied to 107 canonical pulsars with $\\ln \\mathfrak{{B}} \\geq 5$ and all 274 canonical pulsars respectively.",
            label=f"tab:hypermodel_comparison_{param_comb}",
            na_rep="-",
        )


def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument(
        "-c",
        "--config",
        help="Configuration file",
        default="configs/analysis_config.ini",
    )
    aparser.add_argument(
        "-m",
        "--mkcsv",
        help="Create csv file for model comparison",
        action="store_true",
        default=False,
    )
    aparser.add_argument(
        "--only-nonwhite",
        help="limit to only non-white models",
        action="store_true",
        default=False,
    )

    args = aparser.parse_args()

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )
    config.read(args.config)

    parnt_dir_null, child_dir_null, parnt_dirs, child_dirs, if_toas_, save_csvs = (
        get_model_comparison_setup_from_config(config)
    )

    two_cmpnt_nonwhites = []
    two_cmpnt_all = []
    outdir_ref = os.path.join(parnt_dir_null, child_dir_null)
    for pdir, cdir, if_toas, save_csv in zip(
        parnt_dirs, child_dirs, if_toas_, save_csvs
    ):
        outdir = os.path.join(pdir, cdir)
        csv_name = os.path.join(outdir, "model_comparison.csv")
        if args.mkcsv or not os.path.exists(csv_name):
            create_model_data(
                outdir,
                outdir_ref,
                if_toas=if_toas,
                save_csv=save_csv,
            )
        if args.only_nonwhite:
            twocmpnt_nonwhite, utmost_nonwhite = filter_nonwhite(csv_name=csv_name)
        else:
            (
                twocmpnt_nonwhite,
                utmost_nonwhite,
                df_two_cmpnt,
                utmost_nonglitch,
                utmost_nonglitch_nonwhitestrong,
            ) = filter_nonwhite(csv_name=csv_name, return_all=True)
            two_cmpnt_all.append(df_two_cmpnt)
        two_cmpnt_nonwhites.append(twocmpnt_nonwhite)
    if len(two_cmpnt_nonwhites) > 1:
        print("\nDifference of all non-white models:")
        print("1st have but 2nd don't:")
        print(two_cmpnt_nonwhites[0].index.difference(two_cmpnt_nonwhites[1].index))
        print("2nd have but 1st don't:")
        print(two_cmpnt_nonwhites[1].index.difference(two_cmpnt_nonwhites[0].index))

    if args.only_nonwhite:
        create_latex_table_indiv(two_cmpnt_nonwhites[0], utmost_nonwhite)
    else:
        create_latex_table_indiv(
            two_cmpnt_all[0],
            utmost_nonglitch,
            asterisk_objects=twocmpnt_nonwhite.index.difference(
                utmost_nonglitch_nonwhitestrong.index
            ),
            dagger_objects=utmost_nonglitch_nonwhitestrong.index.difference(
                twocmpnt_nonwhite.index
            ),
            glitched_objects=[
                "J0525+1115",
                "J0601-0527",
                "J0659+1414",
                "J0729-1836",
                "J0742-2822",
                "J0758-1528",
                "J0846-3533",
                "J0908-4913",
                "J0922+0638",
                "J1048-5832",
                "J1105-6107",
                "J1123-6259",
                "J1141-3322",
                "J1141-6545",
                "J1257-1027",
                "J1320-5359",
                "J1328-4357",
                "J1452-6036",
                "J1453-6413",
                "J1539-5626",
                "J1644-4559",
                "J1703-4851",
                "J1705-3423",
                "J1720-1633",
                "J1739-2903",
                "J1743-3150",
                "J1757-2421",
                "J1801-0357",
                "J1803-2137",
                "J1818-1422",
                "J1830-1135",
                "J1833-0827",
                "J1836-1008",
                "J1841-0425",
                "J1844-0433",
                "J1852-0635",
                "J1901+0716",
                "J1902+0615",
                "J1909+0007",
                "J1909+1102",
                "J1910+0358",
                "J1910-0309",
                "J1915+1009",
                "J1919+0021",
                "J1926+0431",
                "J2116+1414",
            ],
            remove_objects=[
                "J0820-1350",
                "J1705-1906",
                "J1825-0935",
                "J1835-1020",
                "J1845-0743",
                "J1847-0402",
                "J2346-0609",
            ],  # remove glitching pulsars that not detected by UTMOST
        )
    # create_latex_table_hyper(
    #     outdir,
    #     params=["Qc", "Qs", "tau"],
    #     pars_latex={
    #         "tau": "$\\tau$",
    #         "Qc": "$Q_{\\rm c}$",
    #         "Qs": "$Q_{\\rm s}$",
    #     },
    #     combine=False,
    # )


if __name__ == "__main__":
    main()
