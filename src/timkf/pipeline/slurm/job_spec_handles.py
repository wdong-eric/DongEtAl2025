import os
from ...misc.utils import find_sorted_subdirs, is_incomplete_psr_result


def determine_job_array(outdir_parnt, max_psr_num, model_num=1):
    if not os.path.exists(outdir_parnt):
        return f"1-{max_psr_num}"
    else:
        dirs = find_sorted_subdirs(outdir_parnt)
        incomplete_ids = []
        for dir in dirs:
            if dir.startswith("outdir"):
                prefix, psr_name = dir.split("_")
                task_id = prefix.replace("outdir", "")
                if is_incomplete_psr_result(os.path.join(outdir_parnt, dir), model_num):
                    incomplete_ids.append(task_id)
        array = ",".join(incomplete_ids)
        return array  # can use % to specify ArrayTaskThrottle


def determine_job_runtime():
    """
    TODO: Implement this function
    """
    pass
