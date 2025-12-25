import numpy as np
from mpmath import mp
from ...base import ModelConfig


class WTNPhaseModel:
    """
    White timing noise model for Kalman filter that takes phase data as input.
    This model only fits for deterministic pulsar parameters (freq_dot), EFAC and EQUAD.

    Parameters:
        model_config: ModelConfig object. See models/base.py for details.
        times: Pulsar emission times
        data: phase measurements
        measurement_cov:
            measurement covariance (assumed just take the diagonal elements)
        measurement_matrix:
            measurement matrix. Default is [[1.0, 0.0]] for phase data.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        times,
        data,
        measurement_cov,
        measurement_matrix=mp.matrix([[1.0, 0.0]]),
    ):
        self.times = times
        self.data = data  # phase measurements
        self.R_orig = measurement_cov.copy()
        self.H = measurement_matrix
        self.numeric_impl = model_config.numeric_impl

        if self.numeric_impl != "mpmath":
            raise ValueError("Only mpmath is supported")
        self.n_states = self.H.cols
        self.n_obs_states = self.H.rows

        self.dts = np.diff(self.times)

    def _prepare_KF_essentials(self, params):
        """
        Prepare the Kalman filter essentials: F, Q, B, R, x0, P0.
        """
        self.params = params
        self._initialise_KalmanMatrices()
        self._initial_state_guess()

    def _initial_state_guess(self):
        """
        Initial states and covariances guess for Kalman filter input: x0, P0.
        """
        phic_0 = self.params["phic_0"] * mp.mpf(1)
        omgc_0 = self.params["omgc_0"] * mp.mpf(1)

        Nobs = len(self.times)
        Tobs = self.times[-1] - self.times[0]
        dt_av = Tobs / Nobs
        freq_cov = self.R[0] / (dt_av) ** 2

        self.x0 = mp.matrix([phic_0, omgc_0])
        self.P0 = mp.diag([self.R[0], freq_cov]) * mp.mpf(1)

    def _initialise_KalmanMatrices(self):
        """
        Initialise Kalman matrices: F, Q, B, R.
        """
        self.F = self._construct_transit_F(self.dts)
        self.B = self._construct_control_B(self.dts, self.params["omgc_dot"])
        self.Q = self._construct_proccov_Q(self.dts)
        self.R = self._construct_meascov_R(
            self.R_orig, self.params["EFAC"], self.params["EQUAD"]
        )

    @staticmethod
    def _construct_transit_F(dts):
        F = np.zeros((len(dts), 2, 2), dtype=mp.mpf)
        F[:, 0, 0] = np.ones(len(dts))
        F[:, 0, 1] = dts
        F[:, 1, 0] = np.zeros(len(dts))
        F[:, 1, 1] = np.ones(len(dts))
        return F

    @staticmethod
    def _construct_control_B(dts, omgc_dot):
        B = np.zeros((len(dts), 2), dtype=mp.mpf)
        B[:, 0] = omgc_dot * dts**2 / 2
        B[:, 1] = omgc_dot * dts
        return B

    @staticmethod
    def _construct_proccov_Q(dts):
        Q = np.zeros((len(dts), 2, 2), dtype=mp.mpf)
        return Q

    @staticmethod
    def _construct_meascov_R(R_orig, EFAC, EQUAD):
        return R_orig * EFAC + EQUAD
