This readme is designed to guide users in setting up environments and installing Tempo2 and libstempo on OZSTAR, and submitting jobs. 
The instructions provided here are the step-by-step command I used to set up and are based on ldunn's two reposition, which can be found at the following URL: [ozstar.md](https://gist.github.com/ldunn/49d18951eb86b46dc0035dc9ede459d8) and [tempo2.md](https://gist.github.com/ldunn/04a20627941df1a7f96f6fd5a0a291b9). I also base my instructions on the official documentation of OZSTAR, [here](https://supercomputing.swin.edu.au/docs/).

# Python Environment
- Log in via `ssh <username>@nt.swin.edu.au` (Ngarrgu Tindebeek, "OZSTAR 2") and enter your password (for sure).
- Create a directory like `/fred/oz022/<username>/software` for installing your software, and personally I would also put the vitual environment in this directory. This path will be referred as `$INSTALL_DIR`.
- Create a virtual environment (venv) via `python -m venv /<path_to_new_venv>/<venv_name>`.
- Create a activation shell file `activate.sh`. This shell file will be used to load modules everytime we activate the virtual environment. I'm lazy and just load the modules used for CW, as an example.
```
module load gcc/12.2.0
module load git-lfs/3.2.0
module load cuda/12.0.0
module load python/3.11.2-bare
module load gsl/2.7
module load cfitsio/4.2.0
module load fftw/3.3.10
module load mpfr/4.2.0
module load pgplot/5.2.2
module load openmpi/4.1.5
module load openblas/0.3.21
module load cmake/3.24.3
```
- You may not put shebang `#!/bin/bash` at the start, as it opens a new bash shell to load modules.

I will also export some variables globally in the same shell file (you can chunk it into a different file if you want).
When a variable is exported, it allows other processes to find and use the files/libraries located in our specified directories.
```
INSTALL_DIR=<YOUR_INSTALL_DIR>
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$INSTALL_DIR/lib:$LIBRARY_PATH
export PATH=$INSTALL_DIR/bin:$PATH
export CPATH=$INSTALL_DIR/include:$CPATH
export TEMPO2=$INSTALL_DIR/tempo2/T2runtime
```
<!-- Let me explain some of the above variables:
- LD_LIBRARY_PATH: This variable is used by the system to locate shared libraries at runtime. 
- LIBRARY_PATH: Similar to LD_LIBRARY_PATH, LIBRARY_PATH is used to specify additional directories where libraries are located. 
PATH: The PATH variable contains a list of directories where executable files are located. 
- CPATH: The CPATH variable is used by the C compiler to locate header files.  -->
- The last line will be needed when installing `tempo2`. If you are not using `tempo2`, feel free to omit it.
We need one last piece to activate the venv 
```
source ${INSTALL_DIR}/<venv_name>/bin/activate
```
Now we are ready to activate the venv. We need to run the shell file by calling `source activate.sh`.
The venv is necessary for `pip install`-ing your own packages. For example, if you need to use bilby, just hit `pip install bilby`.

# Tempo2
Clone the `tempo2` [primary repo](https://bitbucket.org/psrsoft/tempo2/src/master/). 
Liam also has his own [fork](https://github.com/ldunn/tempo2/tree/liam_tweak) of `tempo2`, which includes some of his tweaks. 
(**Important**: If you're using Liam's tweak, make sure your file `tempo2/T2runtime/clock/mons2gps.clk` is an actual data file instead of a symlink to a directory you don't have access to).
I'll proceed with the primary repo, as an example.
```
$ cd $INSTALL_DIR
$ git clone https://bitbucket.org/psrsoft/tempo2.git
$ cd tempo2
```
**Important**: Double check you have a line like `export TEMPO2=<path of tempo2>/T2runtime` in `activate.sh` in order to install `tempo2`. If not, add it in and reactivate the venv.

Compling and installing:
```
$ ./bootstrap
$ ./configure --prefix=$INSTALL_DIR
$ make && make install
$ make plugins & make plugins-install
```
Note `--prefix` is to set the path you want to install the binaries and libraries.

# libstempo
Liam strongly recommends using his fork of `libstempo` (as it fixes a uninitialised memory bug), which I shall proceed with.
```
$ git clone https://github.com/ldunn/libstempo.git
$ cd libstempo
$ git checkout ne_sw_ifuncN_segfault_fix
$ pip install -r requirements.txt
$ pip install -e .
```
It seems that there's a bug with missing parameter definition fddmct is ƒixed in the official codes on Dec 6, 2024. If wish to stay with the official version, proceed with (the tempo dependency is bundled in the conda recipe):
```
conda install -c conda-forge libstempo
```

# Submitting jobs
OZSTAR uses slurm as the cluster management and job scheduling system.
See the OZSTAR [manual](https://supercomputing.swin.edu.au/docs/2-ozstar/oz-slurm-create.html) for job creation and submittion (and for any OZSTAR related info). Could also check [slurm](https://slurm.schedmd.com/documentation.html).
- The script `slurm_run_on_DR1.sh` can be used as a template submission script for you to have a look.
- When you submit the script to slurm, simply run `sbatch <your_script_for_submission>.sh`.