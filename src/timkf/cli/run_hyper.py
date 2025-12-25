import os
import configparser
from timkf.hyper_run import (
    construct_hpp_dict_from_config_section,
    get_lnBF_dfs,
    init_argparse,
    gen_tag,
)
from timkf.pipeline import sbatch_temp_job_on_ozstar
from .utils import TMNL_GREEN, TMNL_BLUE, TMNL_RESET

CONFIG_FILE = "configs/hyper_config.ini"


def run_analysis(
    config: configparser.ConfigParser,
    analysis_sect,
    exclude_msp=False,
    exclude_mag=False,
    sampler=None,
):
    print(
        f"\n{TMNL_BLUE}==================== Specified analysis section in {CONFIG_FILE}: {analysis_sect} ===================={TMNL_RESET}"
    )
    print(f"{TMNL_BLUE}========== MSPs excluded: {exclude_msp} =========={TMNL_RESET}")
    print(
        f"{TMNL_BLUE}========== Magnetars excluded: {exclude_mag} =========={TMNL_RESET}"
    )

    hpp_dict = construct_hpp_dict_from_config_section(config, analysis_sect)
    run_script = config["default"].get("run_script")

    for hm_type, hpp in hpp_dict.items():
        print(
            f"{TMNL_GREEN}========== Initiating slurm job for {hm_type} hyper model =========={TMNL_RESET}"
        )
        job_name = f"HRCHY_{analysis_sect}_{hm_type}"
        run_time = config["default"]["job_run_time"]
        cpus_per_task = config["default"].getint("cpus_per_task")
        total_mem = config["default"]["total_mem"]
        only_psrs = config.get(analysis_sect, "only_psrs", fallback=None)
        excl_psrs = config.get(analysis_sect, "excl_psrs", fallback=None)
        if only_psrs is not None:
            only_psrs = [
                psrnm.strip().strip('"').strip("'") for psrnm in only_psrs.split(",")
            ]
        if excl_psrs is not None:
            excl_psrs = [
                psrnm.strip().strip('"').strip("'") for psrnm in excl_psrs.split(",")
            ]

        lnBF_dfs = get_lnBF_dfs(
            hpp_dict,
            BF_threshold=config[analysis_sect].getfloat("lnBF_thres"),
            logger_on=False,
            exclude_msp=exclude_msp,
            exclude_mag=exclude_mag,
            restrict_to_psr=only_psrs,
            exclude_psrs=excl_psrs,
        )

        if sampler is None:
            sampler = config.get("default", "sampler")
        job_name += f"_{sampler}"
        job_name += f"_{len(lnBF_dfs[hm_type])}psrs"
        if exclude_msp:
            job_name += "_exclude_msp"
        if exclude_mag:
            job_name += "_exclude_mag"
        tag = gen_tag(hpp, lnBF_dfs[hm_type], sampler, exclude_msp, exclude_mag)

        outfile_path = os.path.join(hpp.hrchy_outdir, tag, "hrchy.out")
        errfile_path = os.path.join(hpp.hrchy_outdir, tag, "hrchy.err")
        if os.path.exists(os.path.join(hpp.hrchy_outdir, tag, f"{tag}_result.json")):
            print("*_result.json file already exists. Skipping...")
            continue
        # prepare python command
        python_cmd = f"python {run_script} -mtype {hm_type} -sect {analysis_sect}"
        python_cmd += f" -s {sampler}"
        if exclude_msp and exclude_mag:
            python_cmd += " -e msp mag"
        elif exclude_msp:
            python_cmd += " -e msp"
        elif exclude_mag:
            python_cmd += " -e mag"
        if sampler == "ultranest":
            python_cmd = f"mpiexec -np {cpus_per_task} " + python_cmd

        sbatch_temp_job_on_ozstar(
            python_cmd,
            job_name,
            output_file_path=outfile_path,
            error_file_path=errfile_path,
            run_time=run_time,
            cpus_per_task=cpus_per_task,
            total_mem=total_mem,
        )


def main():
    parser = init_argparse()
    args = parser.parse_args()

    config = configparser.ConfigParser(
        allow_no_value=True, interpolation=configparser.ExtendedInterpolation()
    )
    config.read(CONFIG_FILE)

    run_analysis_values = [
        value.strip().strip('"').strip("'")
        for value in config.get("general", "run_analysis").split(",")
    ]
    exclude_msp = "msp" in args.exclude
    exclude_mag = "mag" in args.exclude
    if "all" in run_analysis_values:
        for sect in config.sections():
            if sect.lower().startswith("analysis"):
                run_analysis(config, sect, exclude_msp, exclude_mag, args.sampler)
    else:
        for rav in run_analysis_values:
            run_analysis(config, rav, exclude_msp, exclude_mag, args.sampler)


if __name__ == "__main__":
    main()
