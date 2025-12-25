import numpy as np
import jax
import jax.numpy as jnp
from mpmath import mp
from typing import Union, Literal
from timkf import NP_LONGDOUBLE_TYPE
from .numerics._kalman_mpmath import MPMathKalmanFilterBasics
from .numerics._kalman_numpy import NumpyKalmanFilterBasics
from ..models import TwoComponentPhaseModel, WTNPhaseModel


class KalmanFilterStandard:
    AVAILABLE_NUMERIC_IMPLS = ["numpy", "mpmath", "jax"]

    def __init__(
        self,
        Model: Union[TwoComponentPhaseModel, WTNPhaseModel],
        numeric_impl: Literal["auto", "mpmath", "numpy"] = "auto",
    ):
        """
        Args:
            Model: The model to use for the Kalman filter. It should be a subclass of TwoComponentPhaseModel or WTNPhaseModel.
            numeric_impl: Numeric implementation ('auto', 'mpmath', or 'numpy'). If 'auto', it will inherit the implementation from the model.

        """
        self.Model = Model
        self.data = Model.data[1:]  # the first data point was used as initial state
        if numeric_impl == "auto":
            self.numeric_impl = Model.numeric_impl
        else:
            self.numeric_impl = numeric_impl
        if self.numeric_impl not in self.AVAILABLE_NUMERIC_IMPLS:
            raise ValueError(
                f"Invalid numeric implementation: {self.numeric_impl}. Use 'mpmath' or 'numpy'."
            )

        self.n_states = self.Model.n_states
        self.n_obs_states = self.Model.n_obs_states
        self.n_obs = len(self.data)

        # save Identity matrix
        self.op_dtype = NP_LONGDOUBLE_TYPE  # operation dtype for numpy
        if self.numeric_impl == "numpy":
            self._kf = NumpyKalmanFilterBasics(self.n_obs_states, self.n_states)
        elif self.numeric_impl == "mpmath":
            self._kf = MPMathKalmanFilterBasics(self.n_obs_states, self.n_states)
        elif self.numeric_impl == "jax":
            from .numerics._kalman_jax import JaxKalmanFilterBasics

            self.op_dtype = jnp.float64  # operation dtype for numpy
            self._kf = JaxKalmanFilterBasics(self.n_obs_states, self.n_states)

        self.store_dtype = self._kf.store_dtype
        self.predict = self._kf.predict
        self.update = self._kf.update

    def _ensure_valid_dtype(self):
        if self.numeric_impl == "numpy":
            self.Model.F = np.array(self.Model.F, self.op_dtype)
            self.Model.B = np.array(self.Model.B, self.op_dtype)
            self.Model.Q = np.array(self.Model.Q, self.op_dtype)
            self.Model.R = np.array(self.Model.R, self.op_dtype)
            self.Model.H = np.array(self.Model.H, self.op_dtype).reshape(
                (self.n_obs_states, self.n_states)
            )
            self.Model.x0 = np.array(self.Model.x0, self.op_dtype)
            self.Model.P0 = np.array(self.Model.P0, self.op_dtype).reshape(
                (self.n_states, self.n_states)
            )
            self.data = np.array(self.data, self.op_dtype).reshape(
                (self.n_obs, self.n_obs_states)
            )
        else:
            self.Model.H = mp.matrix(self.Model.H)
            self.Model.x0 = mp.matrix(self.Model.x0)
            self.Model.P0 = mp.matrix(self.Model.P0)
            self.data = mp.matrix(self.data)

    def get_likelihood(self, params, loglikelihood_burn=0) -> np.float64:
        """
        Run the Kalman filter and return the log likelihood. Basically a wrapper
        around the run method with return_states=False.
        """
        self.Model._prepare_KF_essentials(params)
        self._ensure_valid_dtype()

        return self.run(loglikelihood_burn=loglikelihood_burn, return_states=False)

    def get_states(self, params):
        """
        Run the Kalman filter and return the states. Basically a wrapper
        around the run method with return_states=True.
        """
        self.Model._prepare_KF_essentials(params)
        self._ensure_valid_dtype()

        return self.run(loglikelihood_burn=0, return_states=True)

    def run(self, loglikelihood_burn, return_states):
        """
        Run the Kalman filter.
        ==========================
        Parameters
        ----------
        - loglikelihood_burn: number of timesteps to burn for the log likelihood
        - return_states: whether to return the states or not
        Returns
        -------
        - log likelihood or log likelihood and states
        """
        # Initialize the state and covariance
        x = self.Model.x0.copy()
        P = self.Model.P0.copy()

        # Initialize the log likelihood
        lls = np.zeros(self.n_obs, dtype=self.store_dtype)

        if return_states:
            xs = np.zeros((self.n_obs, self.n_states), dtype=self.store_dtype)
            xps = np.zeros((self.n_obs, self.n_states), dtype=self.store_dtype)
            Ps = np.zeros((self.n_obs, self.n_states, self.n_states), self.store_dtype)
            Pps = np.zeros((self.n_obs, self.n_states, self.n_states), self.store_dtype)
            residuals = np.zeros((self.n_obs, self.n_obs_states), self.store_dtype)
            Ss = np.zeros(
                (self.n_obs, self.n_obs_states, self.n_obs_states), self.store_dtype
            )
            Ks = np.zeros(
                (self.n_obs, self.n_states, self.n_obs_states), self.store_dtype
            )

        for timestep in range(0, self.n_obs):
            xp, Pp = self.predict(
                x,
                P,
                self.Model.F[timestep],
                self.Model.B[timestep],
                self.Model.Q[timestep],
            )
            x, P, ll, (res, S, K) = self.update(
                xp, Pp, self.data[timestep], self.Model.H, self.Model.R[timestep + 1]
            )

            lls[timestep] = ll

            if return_states:
                xs[timestep] = np.asarray(x).reshape(self.n_states)
                Ps[timestep] = np.asarray(P).reshape((self.n_states, self.n_states))
                xps[timestep] = np.asarray(xp).reshape(self.n_states)
                Pps[timestep] = np.asarray(Pp).reshape((self.n_states, self.n_states))
                residuals[timestep] = np.asarray(res).reshape(self.n_obs_states)
                Ss[timestep] = np.asarray(S).reshape(
                    (self.n_obs_states, self.n_obs_states)
                )
                Ks[timestep] = np.asarray(K).reshape((self.n_states, self.n_obs_states))

        self.ll_sum = np.sum(lls[loglikelihood_burn:])
        if return_states:
            return self.op_dtype(self.ll_sum), {
                "xs": xs,
                "Ps": Ps,
                "xps": xps,
                "Pps": Pps,
                "lls": lls,
                "residuals": residuals,
                "Ss": Ss,
                "Ks": Ks,
            }
        return self.op_dtype(self.ll_sum)

    def run_jaxscan(self):
        def step(carry, inputs):
            x, P = carry
            F, B, Q, y, H, R = inputs
            # Prediction
            xpred, Ppred = self.predict(x, P, F, B, Q)
            # Update
            x, P, ll, (res, S, K) = self.update(xpred, Ppred, y, H, R)
            return (x, P), (x, P, xpred, Ppred, ll)

        # Stack inputs for lax.scan
        inputs = (
            self.Model.F,
            self.Model.B,
            self.Model.Q,
            self.data,
            self.Model.H,
            self.Model.R[1:],
        )
        carry = (self.Model.x0.copy(), self.Model.P0.copy())
        _, (xreturns, Preturns, xpreds, Ppreds, lls) = jax.lax.scan(step, carry, inputs)

        return lls, xreturns, Preturns, xpreds, Ppreds

    def make_associative_filtering_elements(self):
        # the first data point is set to be the initial guess
        # the first filtering element then should take the 2nd data point (hence the 2nd R matrix)
        first_elems = self._kf.first_filtering_element(
            self.Model.F[0],
            self.Model.B[0],
            self.Model.H,
            self.Model.Q[0],
            self.Model.R[1],
            self.Model.data[1],
            self.Model.x0,
            self.Model.P0,
        )
        # map the rest of len(model.F) - 1 elements
        generic_elems = jax.vmap(
            lambda t: self._kf.generic_filtering_element(
                self.Model.F[t],
                self.Model.B[t],
                self.Model.H,
                self.Model.Q[t],
                self.Model.R[t + 1],
                self.Model.data[t + 1],
            )
        )(jnp.arange(1, len(self.Model.F)).astype(int))
        return tuple(
            jnp.concatenate([jnp.expand_dims(first_e, 0), gen_es])
            for first_e, gen_es in zip(first_elems, generic_elems)
        )

    def run_parallelscan(self, device=None):
        self.Model.H = jnp.array(self.Model.H, self.op_dtype).reshape(
            (self.n_obs_states, self.n_states)
        )
        self.Model.R = jnp.array(self.Model.R, self.op_dtype)
        self.Model.data = jnp.array(self.Model.data, self.op_dtype).reshape(
            (len(self.Model.data), self.n_obs_states)
        )
        set_device(self.Model, device)

        initial_elements = self.make_associative_filtering_elements()
        As, bs, Cs, Js, etas = jax.lax.associative_scan(
            self._kf.filtering_assoc_operator, initial_elements
        )
        return bs, Cs


def set_device(model: TwoComponentPhaseModel, device):
    model.F = jax.device_put(model.F, device=device)
    model.B = jax.device_put(model.B, device=device)
    model.H = jax.device_put(model.H, device=device)
    model.Q = jax.device_put(model.Q, device=device)
    model.R = jax.device_put(model.R, device=device)
    model.data = jax.device_put(model.data, device=device)
    model.x0 = jax.device_put(model.x0, device=device)
    model.P0 = jax.device_put(model.P0, device=device)
