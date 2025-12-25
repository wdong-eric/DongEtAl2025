import os
import subprocess
import tempfile


def on_ozstar_project(project_dir: str = "/fred/oz022/"):
    root_dir = os.getcwd()
    if root_dir.startswith(project_dir):
        return True
    return False


def get_data_path(config, use_data):
    data_path = config[use_data]["my_data_path"]
    return data_path


def get_pipeline_outdir_path(config, use_data):
    outdir_parnt = config[use_data]["outdir_parnt"]
    job_name = config["slurm"]["job_name"]

    pipeline_outdir = os.path.join(outdir_parnt, job_name)
    return pipeline_outdir


def get_outerr_path(parnt_dir, psr_name=None):
    if psr_name:
        out_file_path = os.path.join(
            parnt_dir, f"outdir%a_{psr_name}", f"{psr_name}.out"
        )
        err_file_path = os.path.join(
            parnt_dir, f"outdir%a_{psr_name}", f"{psr_name}.err"
        )
    else:
        out_file_path = os.path.join(parnt_dir, "outdir_log", "DR1_%a.out")
        err_file_path = os.path.join(parnt_dir, "outdir_err", "DR1_%a.err")
    return out_file_path, err_file_path


def get_sbatch_cmd_line(
    job_name,
    run_time,
    out_file_path,
    err_file_path,
    cpus_per_task=1,
    total_mem=None,
    mem_per_cpu=None,
    array=None,
):
    cmd_base = f"sbatch --job-name={job_name} --output={out_file_path} --error={err_file_path} --time={run_time} --cpus-per-task={cpus_per_task}"

    if array:
        cmd_base += f" --array={array}"

    if total_mem and mem_per_cpu:
        raise ValueError("Cannot specify both total_mem and mem_per_cpu")
    if not total_mem and not mem_per_cpu:
        raise ValueError("Must specify either total_mem or mem_per_cpu")
    if total_mem:
        cmd_base += f" --mem={total_mem}"
    if mem_per_cpu:
        cmd_base += f" --mem-per-cpu={mem_per_cpu}"

    return cmd_base


def sbatch_temp_job_on_ozstar(
    python_cmd: str,
    job_name: str,
    output_file_path: str,
    error_file_path: str,
    run_time: str,
    cpus_per_task: int = 1,
    total_mem: str = None,
    mem_per_cpu: str = None,
    array: str = None,
    activate_venv_path: str = "/fred/oz022/wdong/psr_timing/activate_venv.sh",
    mail_type: str = "END,FAIL",
    mail_user: str = None,
):
    if not python_cmd.startswith("python") and not python_cmd.startswith("mpiexec"):
        raise ValueError("python_cmd should start with 'python'")
    if total_mem and mem_per_cpu:
        raise ValueError("Cannot specify both total_mem and mem_per_cpu")
    if not total_mem and not mem_per_cpu:
        raise ValueError("Must specify either total_mem or mem_per_cpu")

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, prefix=f"{job_name}_sbatch_", suffix=".sh"
    ) as tmp_script:
        tmp_script.write("#!/bin/bash\n")
        tmp_script.write(f"#SBATCH --job-name={job_name}\n")
        tmp_script.write(f"#SBATCH --output={output_file_path}\n")
        tmp_script.write(f"#SBATCH --error={error_file_path}\n")
        tmp_script.write(f"#SBATCH --time={run_time}\n")
        tmp_script.write(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
        if array:
            tmp_script.write(f"#SBATCH --array={array}\n")
        if total_mem:
            tmp_script.write(f"#SBATCH --mem={total_mem}\n")
        if mem_per_cpu:
            tmp_script.write(f"#SBATCH --mem-per-cpu={mem_per_cpu}\n")
        if mail_user:
            tmp_script.write(f"#SBATCH --mail-type={mail_type}\n")
            tmp_script.write(f"#SBATCH --mail-user={mail_user}\n")
        tmp_script.write(f"source {activate_venv_path}\n")
        if python_cmd.startswith("mpiexec"):
            tmp_script.write("export OMP_NUM_THREADS=1\n")
            tmp_script.write(
                f"{python_cmd.replace('mpiexec', 'mpiexec --oversubscribe')}\n"
            )
        else:
            tmp_script.write(f"{python_cmd}\n")

        tmp_script_path = tmp_script.name

    job_id = subprocess.run(
        f"sbatch {tmp_script_path}", shell=True, stdout=subprocess.PIPE
    ).stdout.strip()
    print(f"Job submitted with Job ID: {job_id}\n")
    os.remove(tmp_script_path)
