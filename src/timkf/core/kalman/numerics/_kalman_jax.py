import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap


class JaxKalmanFilterBasics:
    """first_filtering_element, generic_filtering_element, and filtering_assoc_operator are included to perform parallel filtering operations with jax.associative_scan."""

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
        self._I_np = jnp.eye(n_states, dtype=jnp.float64)

        self.op_dtype = jnp.float64  # operation dtype for numpy
        self.store_dtype = jnp.float64
        self.predict = self._predict_numpyType
        self.update = self._update_numpyType
        if self.n_obs_states == 1:
            self.update = self._update_numpyType_H1D

    def _predict_numpyType(
        self, x: jnp.array, P: jnp.array, F: jnp.array, B: jnp.array, Q: jnp.array
    ):
        x_predict = F @ x + B
        P_predict = F @ P @ F.T + Q
        return x_predict, P_predict

    def _update_numpyType_H1D(
        self,
        xp: jnp.array,
        Pp: jnp.ndarray,
        y: jnp.array,
        H: jnp.array,
        R: jnp.array,
    ):
        # nonzero indices of H
        nz_idx = jnp.flatnonzero(H)
        # sub-H with nonzero values
        h = H[0][nz_idx]
        P_HT = Pp[:, nz_idx] * h  # save intermediate step
        res = y - xp[nz_idx] * h  # res is a scalar or (1,)
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
        xp: jnp.array,
        Pp: jnp.ndarray,
        y: jnp.array,
        H: jnp.array,
        R: jnp.array,
    ):
        """
        Kalman filter update step.

        RETURNS:
            x_est: [jnp.array]
                estimated state
            P_est: [jnp.array]
                estimated state covariance matrix
            ll: float
                log likelihood of current measurement
        """
        R = jnp.array(R).reshape((self.n_obs_states, self.n_obs_states))
        P_HT = Pp @ H.T  # save intermediate step

        res = y - H @ xp
        S = H @ P_HT + R
        K = P_HT @ jnp.linalg.inv(S)  # float128/longdouble is unsupported in linalg
        # for some reason / S is more numerically stable than * _inv_np(S)
        # K = P_HT / S

        x_est = xp + K @ res
        I_minus_KH = self._I_np - K @ H
        # Joseph formulation of covariance update: ensure symmetry & numerical stable
        P_est = I_minus_KH @ Pp @ I_minus_KH.T + K @ R @ K.T

        ll = self.log_likelihood_numpyType(res, S)
        return x_est, P_est, ll, (res, S, K)

    def log_likelihood_numpyType_scalar(self, res, S):
        log_like = -0.5 * (jnp.log(S) + res**2 / S + jnp.log(2 * jnp.pi))
        # log_like might be a scalar or (1,)
        return jnp.squeeze(log_like)

    def log_likelihood_numpyType(self, res, S):
        log_like = -0.5 * (
            jnp.log(jnp.linalg.det(S))
            + res.T @ jnp.linalg.inv(S) @ res
            + self.n_obs_states * jnp.log(2 * jnp.pi)
        )
        return log_like

    @staticmethod
    def first_filtering_element(F01, B0, H1, Q0, R1, y1, x0, P0):
        F, B, H, Q, R = F01, B0, H1, Q0, R1
        xp1 = F @ x0 + B
        Pp1 = F @ P0 @ F.T + Q
        S1 = H @ Pp1 @ H.T + R
        # direct inverse for computing (K@S).T = P@H.T is less precise than solving for K
        # assume S1,Pp1 symmetric
        K1 = jsp.linalg.solve(S1, H @ Pp1, assume_a="pos").T

        A1 = jnp.zeros_like(F)
        b1 = xp1 + K1 @ (y1 - H @ xp1)
        C1 = Pp1 - K1 @ S1 @ K1.T

        CF, LOW = jsp.linalg.cho_factor(S1)
        # assume S1 symmetric
        eta1 = F.T @ H.T @ jsp.linalg.cho_solve((CF, LOW), y1 - H @ B)
        J1 = F.T @ H.T @ jsp.linalg.cho_solve((CF, LOW), H @ F)
        return A1, b1, C1, J1, eta1

    @staticmethod
    def generic_filtering_element(F, B, H, Q, R, y):
        S = H @ Q @ H.T + R
        CF, LOW = jsp.linalg.cho_factor(S)
        K = jsp.linalg.cho_solve((CF, LOW), H @ Q).T  # assume S,Q symmetric
        I_minus_KH = jnp.eye(F.shape[0]) - K @ H
        res = y - H @ B

        A = I_minus_KH @ F
        b = B + K @ res
        C = I_minus_KH @ Q
        eta = F.T @ H.T @ jsp.linalg.cho_solve((CF, LOW), res)
        J = F.T @ H.T @ jsp.linalg.cho_solve((CF, LOW), H @ F)
        return A, b, C, J, eta

    @staticmethod
    @vmap
    def filtering_assoc_operator(elem_i, elem_j):
        # # note the jsp everywhere
        A1, b1, C1, J1, eta1 = elem_i
        A2, b2, C2, J2, eta2 = elem_j
        dim = A1.shape[0]
        I_eye = jnp.eye(dim)

        I_C1J2 = I_eye + C1 @ J2
        temp = jsp.linalg.solve(I_C1J2.T, A2.T).T
        A = temp @ A1
        b = temp @ (b1 + C1 @ eta2) + b2
        C = temp @ C1 @ A2.T + C2

        I_J2C1 = I_eye + J2 @ C1
        temp = jsp.linalg.solve(I_J2C1.T, A1).T

        eta = temp @ (eta2 - J2 @ b1) + eta1
        J = temp @ J2 @ A1 + J1

        return A, b, C, J, eta
