import numpy as np
from numpy.linalg import inv
from timkf import NP_LONGDOUBLE_TYPE


class NumpyKalmanFilterBasics:
    def __init__(
        self,
        n_obs_states: int,
        n_states: int,
    ):
        """
        Model and data should both use the same dtype.
        """
        self.n_obs_states = n_obs_states

        # save Identity matrix
        self._I_np = np.eye(n_states, dtype=NP_LONGDOUBLE_TYPE)

        self.op_dtype = NP_LONGDOUBLE_TYPE  # operation dtype for numpy
        self.store_dtype = np.float64
        self.predict = self._predict_numpyType
        self.update = self._update_numpyType
        if self.n_obs_states == 1:
            self.update = self._update_numpyType_H1D

    def _predict_numpyType(
        self, x: np.array, P: np.array, F: np.array, B: np.array, Q: np.array
    ):
        """
        Kalman filter prediction step.

        PARAMETERS
        -------------------------------
        - x: [np.array]
            previous state estimate
        - P: [np.array]
            state covariance matrix
        - F: [np.array]
            state transition matrix
        - B: [np.array]

        - Q: [np.array]
            process noise covariance matrix

        RETURNS
        -------------------------------
        - x_predict: [np.array]
            predicted state estimate: F @ x + B
        - P_predict: [np.array]
            predicted state covariance matrix: F @ P @ F.T + Q
        """
        x_predict = F @ x + B
        P_predict = F @ P @ F.T + Q
        return x_predict, P_predict

    def _update_numpyType_H1D(
        self,
        xp: np.array,
        Pp: np.ndarray,
        y: np.array,
        H: np.array,
        R: np.array,
    ):
        # nonzero indices of H
        nz_idx = np.flatnonzero(H)
        # sub-H with nonzero values
        h = H[0][nz_idx]
        P_HT = Pp[:, nz_idx] * h  # save intermediate step
        res = y - xp[nz_idx] * h  # res is a scalar or (1,)
        # y_frac, y_int = np.modf(y)
        # xp_nzi_frac, xp_nzi_int = np.modf(xp[nz_idx])
        # res = (y_frac - xp_nzi_frac) + (y_int - xp_nzi_int)
        S = h @ P_HT[nz_idx] + R  # H @ P_HT + R; S is a scalar or (1, 1)
        # for some reason / S is more numerically stable than * _inv_np(S)
        K = P_HT / S

        x_est = xp + K @ res
        I_minus_KH = self._I_np - K @ H
        # stabilised covariance update: ensure symmetry & numerical stable
        P_est = I_minus_KH @ Pp @ I_minus_KH.T + K @ K.T * R

        ll = self.log_likelihood_numpyType_scalar(res, S)
        return x_est, P_est, ll, (res, S, K)

    def _update_numpyType(
        self,
        xp: np.array,
        Pp: np.ndarray,
        y: np.array,
        H: np.array,
        R: np.array,
    ):
        """
        Kalman filter update step.
        -------------------------------
        - xp: [np.array]
            predicted state estimate
        - Pp: [np.array]
            predicted state covariance matrix
        - y: [np.array]
            measurement
        - H: [np.array]
            measurement matrix
        - R: [np.array]
            measurement noise covariance matrix

        RETURNS
        -------------------------------
        - x_est: [np.array]
            estimated state
        - P_est: [np.array]
            estimated state covariance matrix
        - ll: float
            log likelihood of current measurement
        """
        R = np.array(R).reshape((self.n_obs_states, self.n_obs_states))
        P_HT = Pp @ H.T  # save intermediate step

        res = y - H @ xp
        S = H @ P_HT + R
        K = P_HT @ inv(S)  # float128/longdouble is unsupported in linalg
        # for some reason / S is more numerically stable than * _inv_np(S)
        # K = P_HT / S

        x_est = xp + K @ res
        I_minus_KH = self._I_np - K @ H
        # Joseph formulation of covariance update: ensure symmetry & numerical stable
        P_est = I_minus_KH @ Pp @ I_minus_KH.T + K @ R @ K.T

        ll = self.log_likelihood_numpyType(res, S)
        return x_est, P_est, ll, (res, S, K)

    def log_likelihood_numpyType_scalar(self, res, S):
        log_like = -0.5 * (np.log(S) + res**2 / S + np.log(2 * np.pi))
        # log_like might be a scalar or (1,)
        return np.squeeze(log_like)

    def log_likelihood_numpyType(self, res, S):
        log_like = -0.5 * (
            np.log(np.linalg.det(S))
            + res.T @ inv(S) @ res
            + self.n_obs_states * np.log(2 * np.pi)
        )
        return log_like
