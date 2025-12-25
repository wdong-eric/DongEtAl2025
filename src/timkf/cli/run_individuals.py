import os
import subprocess
import configparser
import argparse
from pathlib import Path

from timkf.pipeline import (
    get_data_path,
    get_pipeline_outdir_path,
    get_outerr_path,
    sbatch_temp_job_on_ozstar,
    determine_job_array,
)
from timkf.misc import find_sorted_subdirs
from .utils import TMNL_GREEN, TMNL_RESET

print("individuals_pipeline.py is at", os.path.dirname(__file__))


def _get_python_cmd(
    config: configparser.ConfigParser,
    SLURM_ARRAY_TASK_ID: int,
    tempo_residuals: bool = False,
):
    """
    TODO: A python version of the shell script
    """
    my_script_path = config.get("general", "my_script_path")
    use_data = config.get("general", "use_data")
    my_data_path = config.get(use_data, "my_data_path")
    outdir_parnt = config.get(use_data, "outdir_parnt")

    job_name = config.get("slurm", "job_name")
    outdir_parnt = os.path.join(outdir_parnt, job_name)
    job_name = f"{use_data}_{job_name}"

    # create a generator to find the first dir that matches the pattern
    # mimic "find | head -n 1" in shell
    existed_outdir = (
        d
        for d in Path(outdir_parnt).rglob(f"outdir{SLURM_ARRAY_TASK_ID}_*")
        if d.is_dir()
    )
    existed_outdir = next(existed_outdir, None)
    if not existed_outdir:
        folder = find_sorted_subdirs(my_data_path, sort_func=lambda x: x)[
            SLURM_ARRAY_TASK_ID - 1
        ]
        psr_name = os.path.basename(os.path.normpath(folder))
    else:
        psr_name = os.path.basename(os.path.normpath(existed_outdir)).split("_")[1]

    python_script = config.get("slurm", "python_script")
    parfile_path = f"{my_data_path}/{psr_name}/{psr_name}.par"
    timfile_path = f"{my_data_path}/{psr_name}/{psr_name}.tim"
    outdirectory = f"{outdir_parnt}/outdir{SLURM_ARRAY_TASK_ID}_{psr_name}/"
    tag = f"{psr_name}"
    nume_impl = config.get("general", "nume_impl")

    # path to the local {}_cfg.ini file in the outdirectory
    local_cfg_path = os.path.join(outdirectory, f"{tag}_cfg.ini")
    if not os.path.exists(local_cfg_path):
        # take the snapshot of the default one, if local cfg.ini not exist
        with open(local_cfg_path, "w") as configfile:
            config.write(configfile)

    python_cmd = f"python {my_script_path}/{python_script} --parfile '{parfile_path}' --timfile '{timfile_path}' --out_directory '{outdirectory}' --tag '{tag}' --nume_impl '{nume_impl}'"
    python_cmd += f" -c '{local_cfg_path}'"

    if tempo_residuals:
        python_cmd += " --tempo_residuals"

    return python_cmd


def main():
    parser = argparse.ArgumentParser(description="Run individuals pipeline")
    parser.add_argument(
        "-c",
        "--cfg_path",
        type=str,
        default="configs/config.ini",
        help="Path to the configuration file (default: configs/config.ini)",
    )
    pargs = parser.parse_args()
    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(), allow_no_value=True
    )
    config.read(pargs.cfg_path)
    # slurm config
    job_name = config["slurm"]["job_name"]
    run_time = config["slurm"]["script_runtime"]
    cpus_per_task = config["slurm"].getint("cpus_per_task")
    total_mem = config["slurm"].get("total_mem")
    mem_per_cpu = config["slurm"].get("mem_per_cpu")

    use_data = config["general"]["use_data"]
    for udata in use_data.split(","):
        data_path = get_data_path(config, udata)
        pipeline_outdir = get_pipeline_outdir_path(config, udata)
        # Determine job array
        count_dirs_in_shell = subprocess.run(
            f"ls -d {data_path}/*/ | wc -l", shell=True, stdout=subprocess.PIPE
        ).stdout.strip()
        num_psr_dirs = int(count_dirs_in_shell)
        run_model_num = config["general"].getint("run_model_num")
        array = determine_job_array(pipeline_outdir, num_psr_dirs, run_model_num)
        # If no directories to process, exit
        if array == "":
            print(f"{TMNL_GREEN}No directories to process. Exiting...{TMNL_RESET}")
            return

        # Generate out, err paths
        out_file_path, err_file_path = get_outerr_path(pipeline_outdir, psr_name=None)

        ids_to_process = (
            range(1, num_psr_dirs + 1)
            if array == f"1-{num_psr_dirs}"
            else [int(id) for id in array.split(",")]
        )
        for array_id in ids_to_process:
            # Generate the python command for each task
            python_cmd = _get_python_cmd(
                config, SLURM_ARRAY_TASK_ID=array_id, tempo_residuals=False
            )
            sbatch_temp_job_on_ozstar(
                python_cmd,
                job_name,
                out_file_path,
                err_file_path,
                run_time,
                cpus_per_task,
                total_mem,
                mem_per_cpu,
                array_id,
            )

        # Print info
        print(f"my_data_path: {data_path}")
        print(f"num_psr_dirs: {num_psr_dirs}")
        print(f"task_array: {array}")
        print(f"pipeline_outdir: {pipeline_outdir}")
        print(f"job_name: {job_name}")
        print(f"out, err paths: {out_file_path, err_file_path}")
        print(f"run_time: {run_time}")
        print(f"cpus_per_task: {cpus_per_task}")
        print(f"total_mem: {total_mem}")
        print(f"mem_per_cpu: {mem_per_cpu}")


if __name__ == "__main__":
    main()
