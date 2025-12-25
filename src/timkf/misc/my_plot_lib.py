import os
import bilby
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from mpmath import mp
from tqdm import tqdm
from corner import corner
import seaborn as sns
from matplotlib import rcParams

from ..constants import DAY_TO_SECONDS, QUANTILE_LEVELS
from ..core import ModelConfig, TwoComponentPhaseModel, KalmanFilterStandard
from . import find_sorted_subdirs, load_psr_bilby_result


def plot_phase_residuals(
    sampler_outdir,
    times,
    phases,
    R_phases,
    omgc_0_par,
    omgc_dot_par,
    result: bilby.core.result.Result = None,
    tempo_psr=None,
):
    data = phases
    phase_errorbars = R_phases ** (1 / 2)
    p, V = np.polyfit(
        np.float64(times),
        np.float64(data),
        2,
        w=(1 / phase_errorbars).astype(np.float64),
        cov=True,
    )  # inverse-variance weighting, w[i] = 1/sigma[i]
    print(f"{omgc_0_par   = }")
    print(f"{omgc_dot_par = }")
    print(f"{'':<20}{p[0] = :>10.3e}")  # quadratic coefficient
    print(f"{'fitted omgc_0:':<20}{p[1] = :>10.3e}")
    print(f"{'fitted omgc_dot:':<18}{2*p[0] = :>10.3e}")
    print(
        f"{'omgc_0_par - fitted omgc_0':<30} = {np.float64(omgc_0_par - p[1]):>10.3e}"
    )
    print(
        f"{'omgc_dot_par - fitted omgc_dot':<30} = {np.float64(omgc_dot_par - 2 * p[0]):>10.3e}"
    )

    nrows = 6
    fig, axs = plt.subplots(nrows, 1, sharex=True, figsize=(3 * nrows, 2 * nrows))
    axs[0].errorbar(
        times / DAY_TO_SECONDS, data, yerr=phase_errorbars, linestyle="", marker="."
    )
    axs[0].set_title("Phase")
    axs[1].errorbar(
        times / DAY_TO_SECONDS,
        data - omgc_0_par * times,
        yerr=phase_errorbars,
        linestyle="",
        marker=".",
    )
    axs[1].set_title("Phase with given linear trend subtracted")
    axs[2].errorbar(
        times / DAY_TO_SECONDS,
        data - omgc_0_par * times - (1 / 2) * omgc_dot_par * times**2,
        yerr=phase_errorbars,
        linestyle="",
        marker=".",
    )
    axs[2].set_title("Phase with given quadratic trend subtracted")
    if tempo_psr is None:
        axs[3].errorbar(
            times / DAY_TO_SECONDS,
            data - p[1] * times - p[0] * times**2,
            yerr=phase_errorbars,
            linestyle="",
            marker=".",
        )
        axs[3].set_title("Phases with fitted quadratic trend subtracted")
    else:
        isort = tempo_psr.pets().argsort()
        data = tempo_psr.phaseresiduals()[isort] * 2 * np.pi
        axs[3].errorbar(
            times / DAY_TO_SECONDS,
            data,
            phase_errorbars,
            linestyle="",
            marker=".",
        )
        axs[3].set_title("Phase residuals returned by tempo2")

    _title_str = "Phase measurements subtracted with Kalman filter"
    if result is not None:
        mle = {
            result.search_parameter_keys[i]: result.samples[-1][i]
            for i in range(len(result.samples[-1]))
        }
        print(f"{mle['omgc_0']   - omgc_0_par   = }")
        print(f"{mle['omgc_dot'] - omgc_dot_par = }")
        measurement_matrix = mp.matrix([[1.0, 0.0, 0.0, 0.0]])

        mod_config = ModelConfig(
            numeric_impl="numpy",
            subtract_linear_trend=True,
            subtract_quadratic_trend=True,
            Omega_ref=omgc_0_par,
            Omega_dot_ref=omgc_dot_par,
        )
        if tempo_psr is not None:
            _title_str = "Tempo2 phase residuals subtracted with Kalman filter"
            mod_config.IS_RESIDUAL = True
        model = TwoComponentPhaseModel(
            mod_config, times, data, R_phases, measurement_matrix
        )
        llsum, states = KalmanFilterStandard(model).get_states(params=mle)
        print("MLE -- log likelihood =", llsum)

        axs[4].errorbar(
            times[1:] / DAY_TO_SECONDS,
            model.data[1:] - states["xps"][:, 0],
            yerr=np.sqrt(R_phases[1:] + states["Pps"][:, 0, 0]),
            linestyle="",
            marker=".",
        )
        axs[5].errorbar(
            times[1:] / DAY_TO_SECONDS,
            model.data[1:] - states["xs"][:, 0],
            yerr=np.sqrt(R_phases[1:] + states["Ps"][:, 0, 0]),
            linestyle="",
            marker=".",
        )
    axs[4].set_title(f"{_title_str} predictions")
    axs[5].set_title(f"{_title_str} estimates")

    for i in range(axs.shape[0]):
        axs[i].set_xlabel("MJD")
        axs[i].label_outer()
        axs[i].set_ylabel("Phase (rad)")
    plt.tight_layout()
    plt.savefig(f"{sampler_outdir}/phases.png")
    plt.close()


def prepare_posteriors_ranges_labels(
    bilby_result: bilby.core.result.Result,
    restrict_to_params: list[str] | None = None,
    weights: np.array = None,
):
    # remove cols that are not useful in posterior df
    posteriors: DataFrame = bilby_result.posterior.copy()
    ignore_keys = ["log_likelihood", "log_prior"] + bilby_result.fixed_parameter_keys
    posteriors.drop(columns=ignore_keys, inplace=True)
    # reweigh posteriors
    if weights is not None:
        posteriors = posteriors.sample(
            n=len(posteriors),
            replace=True,
            weights=weights / np.sum(weights),
            random_state=1,
        )

    corner_ranges = []
    labels_with_unit = []

    for key, label in zip(
        bilby_result.search_parameter_keys, bilby_result.parameter_labels_with_unit
    ):
        if restrict_to_params is not None and key not in restrict_to_params:
            posteriors.pop(key)
            continue
        # convert to log scale
        prior_obj = bilby_result.priors[key]
        corner_ranges.append((prior_obj.minimum, prior_obj.maximum))
        labels_with_unit.append(label)
        if isinstance(prior_obj, bilby.core.prior.Uniform):
            if key in ["omgc_0", "omgc_dot"]:
                corner_ranges[-1] = (posteriors[key].min(), posteriors[key].max())
        # convert values to log scale if needed
        elif isinstance(prior_obj, bilby.core.prior.LogUniform):
            corner_ranges[-1] = np.log10(corner_ranges[-1])
            posteriors[key] = np.log10(posteriors[key])
            # alter the label accordingly
            try:
                latex_label, unit_label = label.split(" [")
                unit_label = " [" + unit_label
            except ValueError:
                latex_label, unit_label = label, ""
            labels_with_unit[-1] = f"$\\log_{{10}}$({latex_label})" + unit_label
        else:
            raise ValueError(
                f"Prior type {type(prior_obj)} not supported in {prepare_posteriors_ranges_labels.__name__}."
            )

    return posteriors, corner_ranges, labels_with_unit


def make_cornerplot(
    result: bilby.core.result.Result,
    restrict_to_params=None,
    save_figname=None,
    **corner_kwargs_input,
):
    samples, corner_range, labels = prepare_posteriors_ranges_labels(
        result, restrict_to_params
    )
    bin_nums = calculate_auto_bin_sizes(samples)
    print("bin_nums =", bin_nums)

    default_corner_kwargs = dict(
        data=samples,
        bins=bin_nums,
        range=corner_range,
        color="tab:purple",
        smooth=None,
        smooth1d=None,
        levels=[1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2)],
        titles=[""] * len(samples.columns),
        show_titles=True,
        title_fmt=".2e",
        title_kwargs={"fontsize": rcParams["axes.titlesize"]},
        quantiles=QUANTILE_LEVELS,
        labels=labels,
    )
    default_corner_kwargs.update(corner_kwargs_input)
    corner_kwargs = default_corner_kwargs
    fig = corner(**corner_kwargs)

    if save_figname:
        plt.savefig(save_figname)
    plt.close()
    return fig, corner_kwargs


def overlay_corner2(
    Kfiltered_data_path_parnt: str = None,
    restrict_to_psr: int | list[str] = None,
    display_progress: bool = False,
    rescale_axes: bool = True,
    restrict_to_params: list[str] = None,
    levels: list[float] = [1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2)],
    display_uniform_prior: bool = False,
    do_kde: bool = False,
    **corner_kwargs,
):
    """
    Overlay corner plots for multiple psrs with results stored in a parent directory.

    Args:
        Kfiltered_data_path_parnt: Path to the parent directory containing PSR subdirectories
        restrict_to_psr: int or list of psr name [str]. If int, the first n PSRs will be plotted
        display_progress: Show progress bar
        rescale_axes: Automatically rescale axes
        restrict_to_params: List of parameters to plot
        levels: Contour levels to plot
        display_uniform_prior: Whether to overplot the uniform priors
        do_kde: Whether to plot KDE instead of histograms
        **corner_kwargs: Additional arguments passed to corner.corner().
            - both_weighted_unweighted, is also accepted an arg.

    Returns:
        Figure
    """
    # use the figure if provided in corner_kwargs
    fig = corner_kwargs.pop("fig", None)

    subdirs = find_sorted_subdirs(
        Kfiltered_data_path_parnt, restrict_to=restrict_to_psr
    )
    cmap = plt.cm.get_cmap("rainbow", len(subdirs))

    psr_iterable = subdirs

    if display_progress:
        psr_iterable = tqdm(psr_iterable, desc="Processing PSRs")

    weights = corner_kwargs.pop("weights", None)
    both_weighted_unweighted = corner_kwargs.pop("both_weighted_unweighted", False)

    for loop_i, subdir in enumerate(psr_iterable):
        prefix, psr_name = subdir.split("_")
        psr_index = int(prefix.replace("outdir", ""))

        result = load_psr_bilby_result(
            os.path.join(Kfiltered_data_path_parnt, subdir), psr_name
        )
        p_samples, ranges, labels = prepare_posteriors_ranges_labels(
            result, restrict_to_params
        )
        p_samples = p_samples.to_numpy()

        bin_sizes = calculate_auto_bin_sizes(p_samples.T)
        wts = _get_weights(weights, psr_name)
        hist_kwargs = _prepare_hist_kwargs(corner_kwargs)
        color = corner_kwargs.pop("colors", cmap(psr_index))
        kde_kwargs = corner_kwargs.pop("kde_kwargs", {})

        if both_weighted_unweighted:
            # also plot the kde plot of unweighted, original samples
            color_unweighted = (
                color if not isinstance(color, list) else color[loop_i + len(subdirs)]
            )
            fig = corner(
                p_samples,
                bins=bin_sizes,
                range=ranges,
                color=color_unweighted,
                levels=levels,
                labels=labels,
                fig=fig,
                weights=None,
                hist_kwargs=hist_kwargs,
                **corner_kwargs,
            )
        fig = corner(
            p_samples,
            bins=bin_sizes,
            range=ranges,
            color=color if not isinstance(color, list) else color[loop_i],
            levels=levels,
            labels=labels,
            fig=fig,
            weights=wts,
            hist_kwargs=hist_kwargs,
            **corner_kwargs,
        )
        if do_kde:
            plot_kdeplot_corner(
                fig,
                p_samples.T,
                len(p_samples[0]),
                ranges,
                weights=wts,
                color=color if not isinstance(color, list) else color[loop_i],
                clear_hist=True,
                **kde_kwargs,
            )
            if both_weighted_unweighted:
                plot_kdeplot_corner(
                    fig,
                    p_samples.T,
                    len(p_samples[0]),
                    ranges,
                    weights=None,
                    color=color_unweighted,
                    clear_hist=True,
                    **kde_kwargs,
                )

    print(
        f"\nAvailable search_parameter_keys: {result.search_parameter_keys}\nPlotted parameters: {restrict_to_params if restrict_to_params is not None else result.search_parameter_keys}"
    )

    dim = len(p_samples[0])
    if display_uniform_prior:
        plot_uniform_priors(fig, ranges, dim)

    if rescale_axes:
        _rescale_axes(fig, dim)

    return fig


def _prepare_hist_kwargs(corner_kwargs: dict):
    hist_kwargs: dict = corner_kwargs.pop("hist_kwargs", {"density": True})
    if (
        corner_kwargs.get("smooth1d") is not None
        and hist_kwargs.get("density") is not None
    ):
        hist_kwargs.pop("density")
    return hist_kwargs


def calculate_auto_bin_sizes(samples: np.ndarray | DataFrame):
    """Calculate the number of bins for each parameter by calling histogram_bin_edges."""
    if isinstance(samples, DataFrame):
        return [
            len(np.histogram_bin_edges(samples[k], "auto")) - 1 for k in samples.columns
        ]
    return [len(np.histogram_bin_edges(s, "auto")) - 1 for s in samples]


def _get_weights(weights, psr_name):
    """Extract weights for the given PSR."""
    if isinstance(weights, np.ndarray):
        assert weights.ndim == 1, (
            "weights should be 1D array for a single PSR; for multiple PSRs, use a dictionary"
        )
        return weights
    elif isinstance(weights, dict) and psr_name in weights:
        return weights[psr_name]
    return None


def _rescale_axes(fig: plt.Figure, dim):
    """Rescale axes of the corner plot."""
    print("Re-autoscaling axes...")
    axes = np.array(fig.axes).reshape((dim, dim))
    for i in np.arange(dim):
        ax_diag = axes[i, i]
        ax_last_row = axes[-1, i]
        ax_last_row.autoscale(axis="y", tight=True)
        try:
            y_max = max([line.get_ydata().max() for line in ax_diag.get_lines()])
            ax_diag.set_ylim(0, y_max)
        except ValueError:
            ax_diag.autoscale(axis="y", tight=True)
        if i == dim - 1:
            ax_diag.autoscale(axis="x", tight=True)


def plot_kdeplot_corner(
    fig: plt.Figure, samples, ndim, ranges, weights=None, clear_hist=False, **kwargs
):
    """Overlay kdeplots (and remove histograms) on a corner plot."""
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        ax = axes[i, i]
        if weights is not None:
            weights = weights / np.sum(weights)
        # clear the histograms
        if clear_hist:
            for artist in ax.get_children():
                if isinstance(artist, plt.Polygon):  # Histograms are Polygon objects
                    artist.remove()
        sns.kdeplot(
            x=samples[i],
            ax=ax,
            weights=weights,
            clip=ranges[i],
            **kwargs,
        )


def plot_uniform_priors(fig: plt.Figure, ranges, ndim: int, **kwargs):
    """Overlay prior ranges on a corner plot.

    Args:
    -----
    - fig: plt.Figure
    - ranges: list of prior ranges
    - ndim: int
    - prior_samples: None. If not None, a region shading the kernel density estimate of the prior samples will be plotted.
    """
    print("Overlaying prior ranges...")
    uniformative_samples = []
    for rmin, rmax in ranges:
        uniformative_samples.append(np.random.uniform(rmin - 0.5, rmax + 0.5, int(1e5)))
    plot_kdeplot_corner(
        fig,
        uniformative_samples,
        ndim,
        ranges,
        color="grey",
        alpha=0.5,
        fill=True,
    )


def add_count_labels(ax: plt.Axes, counts, patches, color="black"):
    """
    Add count labels above the bars in a bar plot.

    Parameters:
    - ax: The axes object of the bar plot.
    - counts: The counts for each bar.
    - patches: The patches (bars) of the bar plot.
    """
    for count, patch in zip(counts, patches):
        if count == 0:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.05,  # vertical offset
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=10,
            color=color,
        )


def plot_medians_hist(
    data: list[np.ndarray],
    ax: plt.Axes,
    bins="auto",
    alpha=0.5,
    label=None,
    xlabel=None,
    ylabel="Counts",
    show_title=True,
    display_counts=True,
    counts_colors=["black"],
    **hist_kwargs,
):
    counts, bins, patches = ax.hist(
        data,
        bins=bins,
        alpha=alpha,
        label=label,
        **hist_kwargs,
    )
    ax.set_ylabel(ylabel)
    ax.legend()
    if xlabel:
        ax.set_xlabel(xlabel)
    if show_title:
        q1, q2, q3 = np.quantile(data, [0.16, 0.5, 0.84])
        ax.set_title(f"{xlabel} = ${q2:.2f}^{{+{q3 - q2:.2f}}}_{{-{q2 - q1:.2f}}}$")
    if display_counts:
        # if patches is a list, it means multiple datasets were plotted
        if isinstance(patches, list):
            # Loop through each dataset and add count labels
            for dataset_idx, (dataset_counts, dataset_patches) in enumerate(
                zip(counts, patches)
            ):
                add_count_labels(
                    ax,
                    dataset_counts,
                    dataset_patches,
                    color=counts_colors[dataset_idx],
                )
        else:
            # Add count labels above bars
            add_count_labels(ax, counts, patches)

    return counts, bins, patches
