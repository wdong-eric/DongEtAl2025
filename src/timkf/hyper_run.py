import os
import configparser
import argparse
import numpy as np
import pandas as pd
from matplotlib import rcParams
import ast
import json

from timkf.hyper import (
    HierarchicalPsrPop,
    HYPER_MODEL_MAPPING_DICT,
    HrchyPopReweightedPosterior,
)
from timkf import src_logger
from timkf.constants import DAY_TO_SECONDS
from timkf.misc import B_surf, is_recycled_millisecond, is_magnetar

CONFIG_FILE = "hyper_config.ini"


def hrchy_corner_range(prior_dict):
    range_list = []
    for key in prior_dict:
        range_list.append((prior_dict[key][0], prior_dict[key][1]))
        range_list.append((0, prior_dict[key][1] - prior_dict[key][0]))
    return range_list


def construct_hpp_dict_from_config_section(
    config: configparser.ConfigParser,
    analysis_sect: str,
    use_sampler: str = "dynesty",
    map_dict: dict = HYPER_MODEL_MAPPING_DICT,
):
    outdir_parnt = config[analysis_sect]["outdir_parnt"]
    outdir_child = config[analysis_sect]["outdir_child"]
    prior_dict = ast.literal_eval(config[analysis_sect]["prior_dict"])
    latex_labels = json.loads(config[analysis_sect]["latex_labels"])
    hyper_model_types = [
        value.strip().strip('"').strip("'")
        for value in config.get(analysis_sect, "hmodel_types").split(",")
    ]
    if "all" in hyper_model_types:
        hyper_model_types = map_dict.keys()

    hpp_dict: dict[str, HierarchicalPsrPop] = {}
    for hm_type in hyper_model_types:
        if hm_type not in map_dict:
            raise ValueError(
                f"Invalid hyper_model_type: {hm_type} in {analysis_sect} section"
            )

        hp_model = map_dict[hm_type](prior_dict, latex_labels)

        hpp_dict[hm_type] = HierarchicalPsrPop(
            os.path.join(outdir_parnt, outdir_child), hp_model, sampler=use_sampler
        )

    src_logger.info(hpp_dict.items())

    return hpp_dict


def get_lnBF_dfs(
    hpp_dict: dict[str, HierarchicalPsrPop],
    BF_threshold: float,
    logger_on=True,
    exclude_msp=False,
    exclude_mag=False,
    restrict_to_psr=None,
    exclude_psrs=None,
):
    lnBF_dfs = {}
    for key, hpp in hpp_dict.items():
        df_two_cmpnt = pd.read_csv(
            os.path.join(hpp.filtered_data_path_parnt, "model_comparison.csv"),
            index_col="PSRJ",
        )

        two_cmpnt_lnBF = df_two_cmpnt[
            df_two_cmpnt["logBF_2Cmpnt_vs_WTNnullHypothesis"] > BF_threshold
        ]
        if exclude_msp:
            msp_names = two_cmpnt_lnBF[
                is_recycled_millisecond(
                    two_cmpnt_lnBF["omgc_0"] / (2 * np.pi),
                    two_cmpnt_lnBF["omgc_dot"] / (2 * np.pi),
                )
            ].index
            two_cmpnt_lnBF = two_cmpnt_lnBF.drop(msp_names)
        if exclude_mag:
            mag_names = two_cmpnt_lnBF[
                is_magnetar(
                    B_surf(two_cmpnt_lnBF["omgc_0"], two_cmpnt_lnBF["omgc_dot"])
                )
            ].index
            two_cmpnt_lnBF = two_cmpnt_lnBF.drop(mag_names)
        if exclude_psrs:
            two_cmpnt_lnBF = two_cmpnt_lnBF.drop(exclude_psrs)
        lnBF_dfs[key] = two_cmpnt_lnBF

    if logger_on:
        src_logger.info(f"Using pulsars with lnBF > {BF_threshold}")
        for key, df in lnBF_dfs.items():
            src_logger.info(df.index)

    if restrict_to_psr is not None:
        for key, df in lnBF_dfs.items():
            psrs_to_be_excluded = df.index.difference(restrict_to_psr)
            lnBF_dfs[key] = df.drop(psrs_to_be_excluded)

    return lnBF_dfs


def get_category_info(lnBF_df: pd.DataFrame, model_type_key: str):
    if "cat" not in model_type_key.lower():
        return None

    cat_df = pd.DataFrame()
    for col in lnBF_df.columns:
        if col.endswith("dist"):
            param = col.split("_")[-1].replace("-dist", "")
            cat_df[f"cat_{param}"] = lnBF_df[col].apply(
                lambda x: 1 if x == "pky" else 0
            )

    return cat_df


def get_Tobs_info(lnBF_df: pd.DataFrame, model_type_key: str):
    if "Tobs" not in model_type_key:
        return None

    Tobs_df = pd.DataFrame()
    Tobs_df["Tobs"] = lnBF_df["Tobs"] * DAY_TO_SECONDS
    return Tobs_df


def init_argparse() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--exclude",
        "-e",
        nargs="*",
        choices=["msp", "mag"],
        default=[],
        help="Exclude MSPs or magnetars",
    )
    argparser.add_argument("--sampler", "-s", type=str, default=None)
    return argparser


def gen_tag(hpp: HierarchicalPsrPop, lnBF_df, sampler, exclude_msp, exclude_mag):
    tag = f"{hpp.hp_model.__class__.__name__}-{len(lnBF_df)}psrs"
    tag += f"_{hpp.sampler}"
    if exclude_msp:
        tag += "_exclude_msp"
    if exclude_mag:
        tag += "_exclude_mag"
    return tag


def main():
    config = configparser.ConfigParser(
        allow_no_value=True, interpolation=configparser.ExtendedInterpolation()
    )
    config.read(CONFIG_FILE)
    sampler = config.get("default", "sampler")

    argparser = init_argparse()
    argparser.add_argument("--model_type", "-mtype", type=str)
    argparser.add_argument("--analysis_sect", "-sect", type=str)
    args = argparser.parse_args()

    sect = args.analysis_sect
    model_type_key = args.model_type
    exclude_msp = "msp" in args.exclude
    exclude_mag = "mag" in args.exclude
    only_psrs = config.get(sect, "only_psrs", fallback=None)
    excl_psrs = config.get(sect, "excl_psrs", fallback=None)
    if only_psrs is not None:
        only_psrs = [
            psrnm.strip().strip('"').strip("'") for psrnm in only_psrs.split(",")
        ]
    if excl_psrs is not None:
        excl_psrs = [
            psrnm.strip().strip('"').strip("'") for psrnm in excl_psrs.split(",")
        ]
    if args.sampler is not None:
        sampler = args.sampler

    hpp_dict = construct_hpp_dict_from_config_section(config, sect, sampler)

    lnBF_dfs = get_lnBF_dfs(
        hpp_dict,
        BF_threshold=config[sect].getfloat("lnBF_thres"),
        logger_on=False,
        exclude_msp=exclude_msp,
        exclude_mag=exclude_mag,
        restrict_to_psr=only_psrs,
        exclude_psrs=excl_psrs,
    )

    hpp = hpp_dict[model_type_key]
    lnBF_df = lnBF_dfs[model_type_key]
    tag = gen_tag(hpp, lnBF_df, sampler, exclude_msp, exclude_mag)

    # log the pulsars used
    src_logger.info(f"Using {len(lnBF_df.index)} pulsars {list(lnBF_df.index)}")

    result_lnBF = hpp(
        restrict_to_psr=lnBF_df.index,
        tag=tag,
        categorical_info=get_category_info(lnBF_df, model_type_key),
        Tobs_info=get_Tobs_info(lnBF_df, model_type_key),
    )

    corner_kwargs = dict(
        show_titles=True,
        title_quantiles=[0.16, 0.5, 0.84],
        # title_fmt='.2f',
        title_kwargs={"fontsize": rcParams["axes.titlesize"]},
    )

    fig_hypermodel = result_lnBF.plot_corner(**corner_kwargs)

    # hprp = HrchyPopReweightedPosterior(
    #     hpp,
    #     hpp_tag=tag,
    #     reweigh_psrs=lnBF_df.index,
    # )


if __name__ == "__main__":
    main()
