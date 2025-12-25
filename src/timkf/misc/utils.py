import os
import numpy as np
import subprocess
import bilby
from tqdm import tqdm
from typing import TypeAlias, Union, Iterator, Callable

# Type aliases for better readability
PSRJ_name: TypeAlias = str


def find_sorted_subdirs(
    outdir, irrelevant_subdirs=None, sort_func: Callable = None, restrict_to=None
) -> list[str]:
    if irrelevant_subdirs is None:
        irrelevant_subdirs = ["outdir_err", "outdir_log", "hrchy"]
    if sort_func is None:

        def sort_func(x: str):
            return x.split("_")[1]

    subdirs = [d for d in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, d))]

    for irsd in irrelevant_subdirs:
        if irsd in subdirs:
            subdirs.remove(irsd)

    if np.isscalar(restrict_to):
        restrict_to = [sort_func(sd) for sd in subdirs][:restrict_to]
    if restrict_to is not None:
        subdirs = [sd for sd in subdirs if sort_func(sd) in restrict_to]

    subdirs.sort(key=sort_func)
    return subdirs


def load_psr_bilby_result(base_path, psr_name):
    """Load the bilby result for the given PSR."""
    return bilby.result.read_in_result(
        os.path.join(base_path, f"{psr_name}_result.json")
    )


# if python sees yield, it will always interpret the function as a generator...
# let's make it explicit and separate the two cases
def gen_load_bilby_results(
    filtered_data_path_parnt,
    tag_extract_func=lambda x: x.split("_")[1],
    restrict_to=None,
) -> Iterator[tuple[PSRJ_name, bilby.result.Result]]:
    """
    Returns:
        generator of (PSRJ, Result) tuples
    """
    subdirs = find_sorted_subdirs(
        filtered_data_path_parnt,
        sort_func=tag_extract_func,
        restrict_to=restrict_to,
    )
    sorted_psr_names = [tag_extract_func(subdir) for subdir in subdirs]
    print("Loading results as generator")
    for psr_name, subdir in tqdm(
        zip(sorted_psr_names, subdirs), desc="Streaming pulsar results"
    ):
        yield (
            psr_name,
            load_psr_bilby_result(
                os.path.join(filtered_data_path_parnt, subdir), psr_name
            ),
        )


def load_bilby_results(
    filtered_data_path_parnt,
    tag_extract_func=lambda x: x.split("_")[1],
    restrict_to=None,
) -> dict[PSRJ_name, bilby.result.Result]:
    subdirs = find_sorted_subdirs(
        filtered_data_path_parnt,
        sort_func=tag_extract_func,
        restrict_to=restrict_to,
    )

    results: dict = {}

    sorted_psr_names = [tag_extract_func(subdir) for subdir in subdirs]
    for psr_name, subdir in tqdm(
        zip(sorted_psr_names, subdirs),
        desc="Loading pulsar results into dictionary",
    ):
        results[psr_name] = load_psr_bilby_result(
            os.path.join(filtered_data_path_parnt, subdir), psr_name
        )
    return results


def update_corner_plots(
    datadir_parnt: str,
    shell_run_python_file: str,
    only_psrs: list[str] = None,
    skip_exists: bool = True,
):
    subdirs = find_sorted_subdirs(datadir_parnt)
    for dir in subdirs:
        prefix, psr_name = dir.split("_")
        if only_psrs is not None and psr_name not in only_psrs:
            continue
        task_id = prefix.replace("outdir", "")

        result_dir = os.path.join(datadir_parnt, dir)

        exist_corner = os.path.exists(
            os.path.join(result_dir, f"{psr_name}_corner.png")
        )
        exist_result = os.path.exists(
            os.path.join(result_dir, f"{psr_name}_result.json")
        )
        if skip_exists and exist_corner:
            continue
        # only run the script if the result file exists but the corner plot does not
        if exist_result:
            print("\nRecreating corner plot for", result_dir)
            cmd = " ".join(
                [
                    f"SLURM_ARRAY_TASK_ID={task_id};",
                    f"source {shell_run_python_file}",
                ]
            )
            subprocess.run(cmd, shell=True)


def is_incomplete_psr_result(outdir: str, model_num):
    result_json_files = [f for f in os.listdir(outdir) if f.endswith("result.json")]
    return len(result_json_files) < model_num
