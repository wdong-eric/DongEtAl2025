"""jax implementation of two-component phase model (numpy -> jax.numpy)"""

import jax
import jax.numpy as jnp
from jax.numpy import polyval
from ....base import ModelConfig, MyBaseModel

jax.config.update("jax_enable_x64", True)


class JaxTwoComponentPhase(MyBaseModel):
    def __init__(self, model_config: ModelConfig):
        self.dtype = jnp.float64
        self.highest_leading_order: int = 3
        # this is set by knowing np.finfo(np.float64).eps ~ 1e-16 and our leading order is x^3
        # because eps is the machine epsilon such that 1 + eps != 1; similar to computing exp - 1
        self.TAYLOR_THRESHOLD = 10 ** jnp.ceil(
            jnp.log10((jnp.finfo(self.dtype).eps) ** (1 / self.highest_leading_order))
        )  # float64 -> 1e-5

        self.Omega_ref = (
            model_config.Omega_ref if model_config.Omega_ref is not None else 0
        )
        self.Omega_dot_ref = model_config.Omega_dot_ref

    @staticmethod
    def param_map(params: dict):
        TAU, LAG, OMGC_DOT = (
            params["tau"],
            params["lag"],
            params["omgc_dot"],
        )
        RATIO = params["ratio"] if "ratio" in params else (1 / params["xs"] - 1)

        taus = (1 + RATIO) * TAU
        tauc = (1 + RATIO) / RATIO * TAU
        Nc = OMGC_DOT + RATIO / (1 + RATIO) * LAG / TAU
        Ns = OMGC_DOT - LAG / TAU * (1 + RATIO) ** -1
        Qc = params["Qc"]
        Qs = (
            params["Qs"]
            if "Qs" in params
            else Qc * ((1 / params["tauprime"] ** 2) - (1 / taus**2)) * tauc**2
        )

        return (
            jnp.float64(tauc),
            jnp.float64(taus),
            jnp.float64(Qc),
            jnp.float64(Qs),
            jnp.float64(Nc),
            jnp.float64(Ns),
        )

    def initialise_KFmatricesFQB(self, params, dts):
        """
        Initialise Kalman matrices: F, Q, B.
        """
        tauc, taus, Qc, Qs, Nc, Ns = self.param_map(params)

        dts = jnp.asarray(dts, dtype=self.dtype)
        self.dts_tau = dts / jnp.float64(params["tau"])
        # jnp.expm1 does not support dtype specification, make sure the input is already in dtype
        self.decay_m1 = jnp.expm1(-self.dts_tau)
        self.decay = self.decay_m1 + 1
        assert (-self.decay_m1 >= 0).all(), (
            "1 - exp_neg_dts_over_tau cannot be negative"
        )
        self.small_mask = jnp.abs(self.dts_tau) < self.TAYLOR_THRESHOLD
        self.small_dts_tau = self.dts_tau[self.small_mask]

        F = self._construct_transit_F(tauc, taus)
        Q = self._construct_proccov_Q(tauc, taus, Qc, Qs)
        B = self._construct_control_B(tauc, taus, Nc, Ns, dts)

        return F, Q, B

    def initial_state_guess(self, params: dict, measurement_R, dt_av):
        """
        Initial states and covariances guess for Kalman filter input: x0, P0.
        """
        tauc, taus, _, _, Nc, Ns = self.param_map(params)
        lag = (tauc * taus / (tauc + taus)) * (Nc - Ns)
        phic_0 = self.dtype(params.get("phic_0"))
        phis_0 = self.dtype(params.get("phis_0"))
        omgc_0 = self.dtype(params["omgc_0"])
        omgs_0 = (
            omgc_0 - lag if params.get("omgs_0", None) is None else params["omgs_0"]
        )
        # when not residualised, Omega_ref = 0
        omgc_0 = omgc_0 - self.Omega_ref
        omgs_0 = omgs_0 - self.Omega_ref

        phase_cov = measurement_R[0]
        freq_cov = phase_cov / (dt_av) ** 2

        X0 = jnp.array([phic_0, phis_0, omgc_0, omgs_0], dtype=self.dtype)
        P0 = jnp.diagflat(
            jnp.array([phase_cov, phase_cov, freq_cov, freq_cov], dtype=self.dtype)
        )

        return X0, P0

    def _construct_proccov_Q(self, tauc, taus, Qc, Qs):
        tau = 1 / (1 / tauc + 1 / taus)
        Qaa = (Qc + Qs) / (tauc + taus) ** 2
        Qab = (Qc * tauc - Qs * taus) / (tauc + taus) ** 2
        Qbb = (Qc * tauc**2 + Qs * taus**2) / (tauc + taus) ** 2

        # assign variable names for convenience (remove self. prefix)
        dts_tau = self.dts_tau
        decay_m1 = self.decay_m1
        decay = self.decay
        decay_sq_m1 = jnp.expm1(-2 * dts_tau)
        decay_m1_sq = decay_m1**2
        leading_2nd_order_decay = decay_m1 + dts_tau
        another_leading_2nd_order_decay = decay_m1 + dts_tau * decay

        small_mask = self.small_mask
        small_dts_tau = self.small_dts_tau
        ## Sig matrices scaled, e.g. by tau^3
        Sig_AA = Qaa * (leading_2nd_order_decay - (1 / 2) * decay_m1_sq)
        # # manual Taylor expansion for small dts/tau - Sig_AA
        TE_COEF = [0, 0, 0, 1 / 3, -1 / 4]
        TE_COEF.reverse()
        Sig_AA = Sig_AA.at[small_mask].set(
            Qaa * polyval(jnp.array(TE_COEF, dtype=self.dtype), small_dts_tau)
        )
        Sig_BB = Qbb * dts_tau**3 / 3
        Sig_AB = Qab * (another_leading_2nd_order_decay + dts_tau**2 / 2)
        # # manual Taylor expansion for small dts/tau - Sig_AB
        TE_COEF = [0, 0, 0, 1 / 3, -1 / 8]
        TE_COEF.reverse()
        Sig_AB = Sig_AB.at[small_mask].set(
            Qab * polyval(jnp.array(TE_COEF, dtype=self.dtype), small_dts_tau)
        )

        ## Sig matrices scaled, e.g. by tau^2
        Sig_Aa = Qaa * decay_m1_sq / 2
        Sig_Ba = -Qab * another_leading_2nd_order_decay
        # # manual Taylor expansion for small dts/tau - Sig_Ba
        TE_COEF = [0, 0, -1 / 2, 1 / 3]
        TE_COEF.reverse()
        Sig_Ba = Sig_Ba.at[small_mask].set(
            -Qab * polyval(jnp.array(TE_COEF, dtype=self.dtype), small_dts_tau)
        )
        Sig_Ab = Qab * leading_2nd_order_decay
        # # manual Taylor expansion for small dts/tau - Sig_Ab
        TE_COEF = [0, 0, 1 / 2, -1 / 6]
        TE_COEF.reverse()
        Sig_Ab = Sig_Ab.at[small_mask].set(
            Qab * polyval(jnp.array(TE_COEF, dtype=self.dtype), small_dts_tau)
        )
        Sig_Bb = Qbb * dts_tau**2 / 2

        ## Sig matrices scaled, e.g. by tau
        Sig_aa = -Qaa * decay_sq_m1 / 2
        Sig_ab = -Qab * decay_m1
        Sig_bb = Qbb * dts_tau

        Q = jnp.empty((len(self.dts_tau), 4, 4))
        # tau**3 terms
        Q = Q.at[:, 0, 0].set((taus**2 * Sig_AA + 2 * taus * Sig_AB + Sig_BB) * tau**3)
        Q = Q.at[:, 0, 1].set(
            (-taus * tauc * Sig_AA + (taus - tauc) * Sig_AB + Sig_BB) * tau**3
        )
        Q = Q.at[:, 1, 1].set((tauc**2 * Sig_AA - 2 * tauc * Sig_AB + Sig_BB) * tau**3)
        Q = Q.at[:, 1, 0].set(Q[:, 0, 1].copy())
        # tau**2 terms
        Q = Q.at[:, 0, 2].set(
            (taus**2 * Sig_Aa + taus * Sig_Ba + taus * Sig_Ab + Sig_Bb) * tau**2
        )
        Q = Q.at[:, 0, 3].set(
            (-taus * tauc * Sig_Aa - tauc * Sig_Ba + taus * Sig_Ab + Sig_Bb) * tau**2
        )
        Q = Q.at[:, 1, 2].set(
            (-taus * tauc * Sig_Aa + taus * Sig_Ba - tauc * Sig_Ab + Sig_Bb) * tau**2
        )
        Q = Q.at[:, 1, 3].set(
            (tauc**2 * Sig_Aa - tauc * Sig_Ba - tauc * Sig_Ab + Sig_Bb) * tau**2
        )
        Q = Q.at[:, 2, 0].set(Q[:, 0, 2].copy())
        Q = Q.at[:, 2, 1].set(Q[:, 1, 2].copy())
        Q = Q.at[:, 3, 0].set(Q[:, 0, 3].copy())
        Q = Q.at[:, 3, 1].set(Q[:, 1, 3].copy())
        # tau terms
        Q = Q.at[:, 2, 2].set((taus**2 * Sig_aa + 2 * taus * Sig_ab + Sig_bb) * tau)
        Q = Q.at[:, 2, 3].set(
            (-tauc * taus * Sig_aa - tauc * Sig_ab + taus * Sig_ab + Sig_bb) * tau
        )
        Q = Q.at[:, 3, 3].set((tauc**2 * Sig_aa - 2 * tauc * Sig_ab + Sig_bb) * tau)
        Q = Q.at[:, 3, 2].set(Q[:, 2, 3].copy())

        return Q

    def _construct_transit_F(self, tauc, taus):
        # dts = np.asarray(dts, dtype=self.dtype)
        tau = 1 / (1 / tauc + 1 / taus)

        # assign variable names for convenience (remove self. prefix)
        dts_tau = self.dts_tau
        decay = self.decay
        decay_m1 = self.decay_m1
        # the 2nd order onward decay term for exp(-dts/tau)
        leading_2nd_order_decay = decay_m1 + dts_tau
        small_mask = self.small_mask
        small_dts_tau = self.small_dts_tau
        # # manual Taylor expansion for small dts/tau -- exp(-dts/tau)-1+dts/tau
        TE_COEF = [0, 0, 1 / 2, -1 / 6]
        TE_COEF.reverse()
        TE_COEF = jnp.array(TE_COEF, dtype=self.dtype)
        # different syntax for jnp array as they are immutable
        leading_2nd_order_decay = leading_2nd_order_decay.at[small_mask].set(
            polyval(TE_COEF, small_dts_tau)
        )
        # leading_2nd_order_decay[small_mask] = polyval(small_dts_tau, TE_COEF)
        assert (leading_2nd_order_decay >= 0).all(), (
            f"exp(-dts/tau) - 1 + dts/tau cannot be negative: {leading_2nd_order_decay}"
        )

        # Initialize F with direct assignments
        F = jnp.empty((len(dts_tau), 4, 4), dtype=self.dtype)
        F = F.at[:, 0, 0].set(1)
        F = F.at[:, 0, 1].set(0)
        F = F.at[:, 0, 2].set(tau**2 * (-decay_m1 / tauc + dts_tau / taus))
        F = F.at[:, 0, 3].set(tau**2 * leading_2nd_order_decay / tauc)

        F = F.at[:, 1, 0].set(0)
        F = F.at[:, 1, 1].set(1)
        F = F.at[:, 1, 2].set(tau**2 * leading_2nd_order_decay / taus)
        F = F.at[:, 1, 3].set(tau**2 * (-decay_m1 / taus + dts_tau / tauc))

        F = F.at[:, 2, 0].set(0)
        F = F.at[:, 2, 1].set(0)
        F = F.at[:, 2, 2].set((tauc + taus * decay) / (tauc + taus))
        F = F.at[:, 2, 3].set(-tau * decay_m1 / tauc)

        F = F.at[:, 3, 0].set(0)
        F = F.at[:, 3, 1].set(0)
        F = F.at[:, 3, 2].set(-tau * decay_m1 / taus)
        F = F.at[:, 3, 3].set((taus + tauc * decay) / (tauc + taus))

        return F

    def _construct_control_B(self, tauc, taus, Nc, Ns, dts):
        tau = (tauc * taus) / (tauc + taus)
        Na = (Nc - Ns) / (tauc + taus)
        Nb = (tauc * Nc + taus * Ns) / (tauc + taus)
        if self.Omega_dot_ref is not None:
            Nb = Nb - self.Omega_dot_ref

        # assign variable names for convenience (remove self. prefix)
        dts_tau = self.dts_tau
        decay_m1 = self.decay_m1
        Nb_dts = Nb * dts
        Nb_dtsSq_half = Nb * dts**2 / 2
        Na_tau_1m_dts_over_tau_decay = Na * tau * (-decay_m1)
        # the 2nd order onward decay term for exp(-dts/tau)
        leading_2nd_order_decay = decay_m1 + dts_tau
        # # manual Taylor expansion for small dts/tau -- exp(-dts/tau)-1+dts/tau
        small_mask = self.small_mask
        small_dts_tau = self.small_dts_tau
        TE_COEF = [0, 0, 1 / 2, -1 / 6]
        TE_COEF.reverse()
        leading_2nd_order_decay = leading_2nd_order_decay.at[small_mask].set(
            polyval(jnp.array(TE_COEF, dtype=self.dtype), small_dts_tau)
        )
        assert (leading_2nd_order_decay >= 0).all(), (
            f"exp(-dts/tau) - 1 + dts/tau cannot be negative: {leading_2nd_order_decay}"
        )
        Na_tauSq_ = Na * tau**2 * leading_2nd_order_decay

        B = jnp.zeros((len(dts_tau), 4))
        # Assign values to B matrix
        B = B.at[:, 0].set(taus * Na_tauSq_ + Nb_dtsSq_half)
        B = B.at[:, 1].set(-tauc * Na_tauSq_ + Nb_dtsSq_half)
        B = B.at[:, 2].set(taus * Na_tau_1m_dts_over_tau_decay + Nb_dts)
        B = B.at[:, 3].set(-tauc * Na_tau_1m_dts_over_tau_decay + Nb_dts)
        return B

    def _construct_meascov_R(self, R_orig, EFAC, EQUAD):
        return R_orig * EFAC + EQUAD
