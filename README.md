# Pulsar Population Analysis and Kalman Filtering

## Table of Contents

- [About](#about)
- [Important Notes](#important-notes)
- [Key Components](#key_components)
- [Basic Usage](#basic_Usage)
- [Getting Started](#getting_started)
- [TODO](#todo)

## About

## Important Notes
- This repo contains details relevant to the paper [*Measuring the crust-superfluid coupling time-scale for 105 UTMOST pulsars with a Kalman filter*](https://doi.org/10.1093/mnras/staf2053). Note that the results in the paper were generated using the `mpmath` implementation due to historical issues with precision. A faster `numpy` version that handles the precision internally by looking at residuals is also provided (and lightly-tested).

## Key Components in [`src/timkf`](src/timkf) <a name = "key_components"></a>

### Core
The [`core`](src/timkf/core/) module contains the core methods for analyzing per-pulsar level pulsar timing data, focusing on relevant models and the Kalman filtering implementation. It provides modern numerical implementations.
- **Kalman Filtering** in [`core/kalman`](src/timkf/core/kalman/__init__.py). The major API function is [`KalmanFilterStandard`](src/timkf/core/kalman/kalman.py) with numerical implementations using [`mpmath`](src/timkf/core/kalman/numerics/_kalman_mpmath.py) (user-specified arbitrary precision) and [`numpy`](src/timkf/core/kalman/numerics/_kalman_numpy.py) (~20x faster).
- **Physical Models**: Modular implementations of pulsar timing models in [`core/models`](src/core/models/__init__.py). Currently, we only have two models acting on phase data: 
    - one-component [`WTNPhaseModel`](src/timkf/core/models/one_component/phase/wtn.py), and 
    - two-component [`TwoComponentPhaseModel`](src/timkf/core/models/two_component/phase/phases.py). The two-component model has also two numerical implementations [`mpmath`](src/timkf/core/models/two_component/phase/numerics/mpmath_impl.py) and [`numpy`](src/timkf/core/models/two_component/phase/numerics/numpy_impl.py).
- **Sampling with bilby**: [`sampling.py`](src/timkf/core/pipeline/sampling.py) defines a few likelihood classes (e.g., `BaseKFLikelihood`, `TwoCompPhaseKFLikelihood`, etc.) that inherit `bilby.Likelihood` and wraps our own likelihood method for using `bilby` as a front end for nested sampling.


### Pipeline
Some useful methods for analysis workflows are stored in [`pipeline`](src/timkf/pipeline/)
- **Main Pipeline Infrastructure**:
    - [`cli.py`](src/timkf/pipeline/cli.py) defines a `PipelineCLI` class that serves as a command line interfacing for handling command line inputs.
    - [`data_handling.py`](src/timkf/pipeline/data_handling.py) defines a `PulsarDataLoader` class that reads .par and .tim files, and returns data/objects that will be used in the analysis.
    - [`prior_handling.py`](src/timkf/pipeline/prior_handling.py) contains a `create_PriorDict` method that takes a form of a dictionary for parameter prior distributions and generates the corresponding Bilby `PriorDict` object. Relevant LATEX_LABELS and UNIT_LABELS are defined in the same file.
    - [`analysis_core.py`](src/timkf/pipeline/analysis_core.py) defines a `AnalysisRunner` class that provides a relatively simple way of creating KF likelihood and running Bilby samplers with pre-defined arguments in `.ini` config file.
- **Slurm**: The [`pipeline/slurm`](src/timkf/pipeline/slurm/) module contains methods that are useful for submitting jobs on HPC systems like OzStar.
    - e.g., `sbatch_temp_job_on_ozstar(*args)` allows `sbatch` submission of a slurm job through a python function on OzStar.

### Hyper
The [`hyper`](src/timkf/hyper) module contains methods for hierarchical Bayesian inference.

### Command Line Tools
The [`cli`](src/timkf/cli/) directory contains modules that define some convenient command line tools, such as
- `timkf-indivs`: Run the inference pipeline on individual pulsars.
- `timkf-hyper`: Run the inference pipeline for the hypermodel analysis.
- `timkf-corner`: Generate corner plots for individual pulsars with selected parameters or for hierarchical results. Supports overlaying multiple plots for comparison.
- `timkf-modcmp`: Produce CSV files and LaTeX tables summarizing inference results. Includes publication-ready latex tables for canonical pulsars, recycled pulsars, magnetars, and the hypermodel.
Run `--help` for each command to show help message
```bash
timkf-indivs --help
timkf-hyper --help
timkf-corner --help
timkf-modcmp --help
```


## Basic Usage <a name = "basic_Usage"></a>
A pre-specified `prior_dict_input` that defines the prior distributions is needed.
If wanted more freedom:
```python
from timkf.core import ModelConfig, TwoComponentPhaseModel
from timkf.pipeline import PulsarDataLoader, PipelineCLI, AnalysisRunner

# defines some command line arguments that allows you to pass .par .tim files
prsrargs = PipelineCLI().args
# load the processed pulsar data from par and tim file; dtype specifies the returned type
(times, phases, R_phase), (omgc_0_par, omgc_dot_par), _ = PulsarDataLoader(dtype=).load(prsrargs.parfile, prsrargs.timfile)
# Initialise the AnalysisRunner object
arunner = AnalysisRunner(src_config)
# Initialise priors
arunner.setup_priors(prior_dict_input, *args)
# create a model configuration, for example
model_config = ModelConfig(
    numeric_impl=prsrargs.nume_impl,  # numpy/mpmath
    IS_RESIDUAL=prsrargs.tempo_residuals,  # whether the data is already residualised, e.g. through tempo2
    subtract_linear_trend=True,
    Omega_ref=omgc_0_par,
    subtract_quadratic_trend=True,
    Omega_dot_ref=omgc_dot_par,
)
# pass the model_config to a model you want
model = TwoComponentPhaseModel(model_config, *args)
arunner.setup(model)
# run sampling
result = arunner.run_sampling(sampler_outdir, sampler_tag)
```
<!-- Run full analysis pipeline
pipeline = AnalysisPipeline(model)
results = pipeline.run() -->

## Getting Started <a name = "getting_started"></a>

### Requirements
See [`pyproject.toml`](pyproject.toml) or [`requirements.txt`](requirements.txt) for Python dependencies. Packages like `ultranest`, `mpi4py`, `jax`, etc. are not mandatory for core usage, unless wanted.

### 🔧 Running the tests <a name = "tests"></a>

Tests are rough and automation is not well-implemented. (They are mostly for my own heuristic testing, sorry...)

<!--#### Break down into end-to-end tests

Explain what these tests test and why

```
Give an example
``` -->

<!--#### And coding style tests

Explain what these tests test and why

```
Give an example
``` -->

## Authors <a name = "authors"></a>

- [@wdong-eric](https://github.com/wdong-eric/PulsarTiming-population)

## TODO
- `tests` refactoring.
- `timkf/hyper` refactoring.
- Include dispersion measure handling.