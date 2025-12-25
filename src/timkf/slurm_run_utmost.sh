#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=12:00:00

# Function to check if a variable is defined
check_var_defined() {
    if [ -z "${!1}" ]; then
        echo "Error: Variable $1 is not defined."
        exit 1
    else
        echo "$1 is defined: ${!1}"
    fi
}

source /fred/oz022/wdong/psr_timing/activate_venv.sh

config_file="/fred/oz022/wdong/psr_timing/src/timkf/config.ini"
my_script_path=$(crudini --get $config_file general my_script_path)
use_data=$(crudini --get $config_file general use_data)
nume_impl=$(crudini --get $config_file general nume_impl)
my_data_path=$(crudini --get $config_file $use_data my_data_path)
outdir_parnt=$(crudini --get $config_file $use_data outdir_parnt)

job_name=$(crudini --get $config_file slurm job_name)
outdir_parnt="$outdir_parnt/$job_name/"
python_script=$(crudini --get $config_file slurm python_script)

# Remove section specifier like ${general:<>} in .ini and substitute my_script_path back in
export my_script_path
job_name=${use_data}-${job_name}
my_data_path=$(echo $my_data_path | sed 's/{[^:]*://g' | sed 's/}//g' | envsubst)
outdir_parnt=$(echo $outdir_parnt | sed 's/{[^:]*://g' | sed 's/}//g' | envsubst)

# Check if all required variables are defined
check_var_defined "my_script_path"
check_var_defined "python_script"
check_var_defined "my_data_path"
check_var_defined "outdir_parnt"
echo "Current Array Task ID: ${SLURM_ARRAY_TASK_ID}"

existed_outdir=$(find "$outdir_parnt" -type d -name "outdir${SLURM_ARRAY_TASK_ID}_*" | head -n 1)

if [ -z "$existed_outdir" ]; then
    echo "No matching existed outdir found for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $outdir_parnt; Creating a new outdir..."
    # Access the par and tim files in the a-th subfolder of data_path
    # -n tells sed to suppress unnecessary printing, '${a}p' tells sed to print the a-th line
    folder=$(ls -d ${my_data_path}/*/ | sed -n "${SLURM_ARRAY_TASK_ID}p") 
    psr_name=$(basename "$folder")
else
    echo "Matching existed outdir found for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $outdir_parnt; Extracting the pulsar name from the existed outdir..."
    psr_name=$(basename "$existed_outdir" | cut -d'_' -f2)
fi
check_var_defined "psr_name"

parfile_path="${my_data_path}/${psr_name}/${psr_name}.par"
timfile_path="${my_data_path}/${psr_name}/${psr_name}.tim"
outdir_child="$outdir_parnt/outdir${SLURM_ARRAY_TASK_ID}_${psr_name}/"
tag="${psr_name}"

# quote the vars to ensure shell interpret them as single arg.
python ${my_script_path}/${python_script} --parfile "$parfile_path" --timfile "$timfile_path" --out_directory "$outdir_child" --tag "$tag" --nume_impl "$nume_impl" --tempo_residuals

echo "${SLURM_ARRAY_TASK_ID} - ${psr_name} END."