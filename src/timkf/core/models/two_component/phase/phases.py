import numpy as np
import mpmath as mp

from ...base import ModelConfig
from .numerics.numpy_impl import NumpyTwoComponentPhase
from .numerics.mpmath_impl import MPMathTwoComponentPhase


class TwoComponentPhaseModel:
    """
    Two-component timing model for pulsars with phase data.

    Parameters:
        model_config: ModelConfig object. See models/base.py for details.
        times: Pulsar emission times
        data: phase relevant data
        measurement_cov:
            measurement covariance (assumed just take the diagonal elements)
        measurement_matrix:
            measurement matrix. Default is [[1.0, 0.0, 0.0, 0.0]] for phase data.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        times,
        data,
        measurement_cov,
        measurement_matrix=mp.matrix([[1.0, 0.0, 0.0, 0.0]]),
    ):
        # setup the model config
        self.config = model_config
        self.numeric_impl = model_config.numeric_impl
        if model_config.numeric_impl == "auto":
            self.numeric_impl = "mpmath" if isinstance(times[0], mp.mpf) else "numpy"
        # assign the numeric implementation for the model
        if self.numeric_impl == "mpmath":
            self.impl = MPMathTwoComponentPhase(model_config)
        elif self.numeric_impl == "numpy":
            self.impl = NumpyTwoComponentPhase(model_config)
        elif self.numeric_impl == "jax":
            from .numerics.jax_impl import JaxTwoComponentPhase

            self.impl = JaxTwoComponentPhase(model_config)

        # store the model parameters
        self.times = times
        self.data = data  # phase/phase residual measurements
        self.R_orig = measurement_cov.copy()
        self.H = measurement_matrix
        self.dts = np.diff(self.times)
        self.T_obs = self.times[-1] - self.times[0]
        print(
            f"dts: non-zero min {self.dts[self.dts != 0].min()}, max {self.dts.max()}, T_obs {self.T_obs}"
        )
        # store useful dimensions
        self.n_states = self.H.cols
        self.n_obs_states = self.H.rows
        self.n_obs = len(self.times)
        # only subtract linear/quadratic trend if data is already residualised
        if not self.config.IS_RESIDUAL and (
            self.config.subtract_linear_trend or self.config.subtract_quadratic_trend
        ):
            self._residualise_data()

        self._ensure_valid_dtype()

    def _prepare_KF_essentials(self, params: dict):
        """
        Prepare the Kalman filter essentials: F, Q, B, R, x0, P0.
        """
        self.F, self.Q, self.B = self.impl.initialise_KFmatricesFQB(params, self.dts)
        self.R = self.impl._construct_meascov_R(
            self.R_orig, params["EFAC"], params["EQUAD"]
        )

        dt_av = self.T_obs / self.n_obs
        # Might be a good idea to use phic_0 = phis_0 = self.data[0] as initial estimates.
        if params.get("phic_0", None) is None:
            params["phic_0"] = self.data[0]
        if params.get("phis_0", None) is None:
            params["phis_0"] = self.data[0]
        self.x0, self.P0 = self.impl.initial_state_guess(params, self.R, dt_av)

    def _residualise_data(self):
        """
        Residualise the data by subtracting Omega_ref or/and Omega_dot_ref given in the config.
        """
        info_str = "Using phase residuals for the two-component model: phi_res = phi - Omega_ref * t"
        self.data = self.data - self.config.Omega_ref * self.times

        if self.config.subtract_quadratic_trend:
            self.data = self.data - self.config.Omega_dot_ref * self.times**2 / 2
            print(info_str + " - Omega_dot_ref * t^2 / 2")
        else:
            print(info_str)

    def _ensure_valid_dtype(self):
        """
        Ensure that the numeric implementation is consistent with the data type.
        """
        if self.numeric_impl == "numpy":
            self.R_orig = np.array(self.R_orig, dtype=np.float64)
