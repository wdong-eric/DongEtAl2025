module load gcc/12.2.0
# module load cuda/12.0.0
module load python/3.11.2-bare
module load pgplot/5.2.2
module load openmpi/4.1.4

INSTALL_DIR="/fred/oz022/wdong/software"
export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=${INSTALL_DIR}/lib:$LIBRARY_PATH
export PATH=${INSTALL_DIR}/bin:$PATH
export CPATH=${INSTALL_DIR}/include:$CPATH
export TEMPO2=${INSTALL_DIR}/tempo2/T2runtime

# VENV_PATH="/fred/oz022/wdong/software/psrtim_venv"
source ${INSTALL_DIR}/psrtim_venv/bin/activate

# add dir path to the python path, so module packages can be imported in python
export PYTHONPATH=$PYTHONPATH:/fred/oz022/wdong/psr_timing