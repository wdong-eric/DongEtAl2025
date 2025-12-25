import numpy as np
from mpmath import mp
from typing import Literal
from timkf import NP_LONGDOUBLE_TYPE


class MPMathKalmanFilterBasics:
    def __init__(
        self,
        n_obs_states: int,
        n_states: int,
    ):
        """
        Basic Kalman filter prediction and update steps, and log likelihood
        calculation using mpmath implementation.
        """
        self.n_obs_states = n_obs_states

        # save Identity matrix
        self._I_mp = mp.eye(n_states)

        self.op_dtype = NP_LONGDOUBLE_TYPE  # operation dtype for numpy
        self.store_dtype = mp.mpf
        self.predict = self._predict_mpType
        self.update = self._update_mpType
        if self.n_obs_states == 1:
            self.update = self._update_mpType_H1D

    def _predict_mpType(
        self,
        x: Literal["mp.matrix"],
        P: Literal["mp.matrix"],
        F: Literal["mp.matrix"],
        B: Literal["mp.matrix"],
        Q: Literal["mp.matrix"],
    ):
        """
        Kalman filter prediction step.

        PARAMETERS
        -------------------------------
        - x: [mp.matrix]
            previous state estimate
        - P: [mp.matrix]
            state covariance matrix
        - F: [mp.matrix]
            state transition matrix
        - B: [mp.matrix]

        - Q: [mp.matrix]
            process noise covariance matrix

        RETURNS
        -------------------------------
        - x_predict: [mp.matrix]
            predicted state estimate: F * x + B
        - P_predict: [mp.matrix]
            predicted state covariance matrix: F * P * F.T + Q
        """
        F = mp.matrix(F)
        B = mp.matrix(B)
        Q = mp.matrix(Q)
        x_predict = F * x + B
        P_predict = F * P * F.T + Q
        return x_predict, P_predict

    def _update_mpType_H1D(
        self,
        xp: Literal["mp.matrix"],
        Pp: Literal["mp.matrix"],
        y: Literal["mp.matrix"],
        H: Literal["mp.matrix"],
        R: Literal["mp.matrix"],
    ):
        # nonzero indices of H
        nz_idx = int(np.flatnonzero(H)[0])
        # sub-H with nonzero values
        h = H[nz_idx]
        P_HT = Pp[:, nz_idx]  # save intermediate step
        res = y - xp[nz_idx]  # res is a scalar or (1,)
        S = h * P_HT[nz_idx] * h + R  # H @ P_HT + R; S is a scalar or (1, 1)
        K = P_HT / S

        x_est = xp + K * res
        # need high precision, matrix supposed to be symmetrical
        I_minus_KH = self._I_mp.copy()
        I_minus_KH[:, nz_idx] = -K  # - K * H
        I_minus_KH[nz_idx, nz_idx] = mp.mpf(1) - K[nz_idx]

        P_est = I_minus_KH * Pp
        # can use numpy version for likelihood, no precision issues
        ll = self.log_likelihood_numpyType_scalar(
            self.op_dtype(res), np.array(S, dtype=self.op_dtype)
        )
        return x_est, P_est, ll, (res, S, K)

    def _update_mpType(
        self,
        xp: Literal["mp.matrix"],
        Pp: Literal["mp.matrix"],
        y: Literal["mp.matrix"],
        H: Literal["mp.matrix"],
        R: Literal["mp.matrix"],
    ):
        """
        Kalman filter update step.
        -------------------------------
        - xp: [mp.matrix]
            predicted state estimate
        - Pp: [mp.matrix]
            predicted state covariance matrix
        - y: [mp.matrix]
            measurement
        - H: [mp.matrix]
            measurement matrix
        - R: [mp.matrix]
            measurement noise covariance matrix

        RETURNS
        -------------------------------
        - x_est: [mp.matrix]
            estimated state
        - P_est: [mp.matrix]
            estimated state covariance matrix
        - ll: float
            log likelihood of current measurement
        """
        P_HT = Pp * H.T  # save intermediate step

        res = y - H * xp  # measurement residual (innovation)
        S = H * P_HT + R  # measurement residual covariance
        K = P_HT * S ** (-1)  # Kalman gain
        x_est = xp + K * res
        # need high precision, matrix supposed to be symmetrical
        P_est = (self._I_mp - K * H) * Pp
        ll = self.log_likelihood_mpType(res, S)
        return x_est, P_est, ll, (res, S, K)

    def log_likelihood_mpType_scalar(self, res, S):
        ll = -0.5 * (mp.log(S) + res**2 / S + mp.log(2 * mp.pi))
        return ll

    def log_likelihood_mpType(self, res, S):
        log_like = -0.5 * (
            mp.log(mp.det(S))
            + res.T * S**-1 * res  # squared Mahalanobis distance
            + self.n_obs_states * mp.log(2 * mp.pi)
        )
        return log_like[0]

    def log_likelihood_numpyType_scalar(self, res, S):
        log_like = -0.5 * (np.log(S) + res**2 / S + np.log(2 * np.pi))
        return log_like
