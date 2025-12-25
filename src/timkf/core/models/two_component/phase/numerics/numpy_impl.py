"""numpy implementation of two-component phase model with numerical stability controls"""

import numpy as np
from numpy.polynomial.polynomial import polyval
from timkf import NP_LONGDOUBLE_TYPE
from ....base import ModelConfig, MyBaseModel


class NumpyTwoComponentPhase(MyBaseModel):
    def __init__(self, model_config: ModelConfig):
        self.dtype = NP_LONGDOUBLE_TYPE
        self.highest_leading_order: int = 3
        # this is set by knowing np.finfo(np.float64).eps ~ 1e-16 and our leading order is x^3
        # because eps is the machine epsilon such that 1 + eps != 1; similar to computing exp - 1
        self.TAYLOR_THRESHOLD = 10 ** np.ceil(
            np.log10((np.finfo(self.dtype).eps) ** (1 / self.highest_leading_order))
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
            NP_LONGDOUBLE_TYPE(tauc),
            NP_LONGDOUBLE_TYPE(taus),
            NP_LONGDOUBLE_TYPE(Qc),
            NP_LONGDOUBLE_TYPE(Qs),
            NP_LONGDOUBLE_TYPE(Nc),
            NP_LONGDOUBLE_TYPE(Ns),
        )

    def initialise_KFmatricesFQB(self, params, dts):
        """
        Initialise Kalman matrices: F, Q, B.
        """
        tauc, taus, Qc, Qs, Nc, Ns = self.param_map(params)

        dts = np.asarray(dts, dtype=self.dtype)
        self.dts_tau = dts / NP_LONGDOUBLE_TYPE(params["tau"])
        self.decay_m1 = np.expm1(-self.dts_tau, dtype=self.dtype)
        self.decay = self.decay_m1 + 1
        assert (-self.decay_m1 >= 0).all(), (
            "1 - exp_neg_dts_over_tau cannot be negative"
        )
        self.small_mask = np.abs(self.dts_tau) < self.TAYLOR_THRESHOLD
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

        X0 = np.array([phic_0, phis_0, omgc_0, omgs_0], dtype=self.dtype)
        P0 = np.diagflat([phase_cov, phase_cov, freq_cov, freq_cov]).astype(self.dtype)

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
        decay_sq_m1 = np.expm1(-2 * dts_tau, dtype=self.dtype)
        decay_m1_sq = decay_m1**2
        leading_2nd_order_decay = decay_m1 + dts_tau
        another_leading_2nd_order_decay = decay_m1 + dts_tau * decay

        small_mask = self.small_mask
        small_dts_tau = self.small_dts_tau
        ## Sig matrices scaled, e.g. by tau^3
        Sig_AA = Qaa * (leading_2nd_order_decay - (1 / 2) * decay_m1_sq)
        # # manual Taylor expansion for small dts/tau - Sig_AA
        TE_COEF = [0, 0, 0, 1 / 3, -1 / 4]
        Sig_AA[small_mask] = Qaa * polyval(small_dts_tau, TE_COEF)
        Sig_BB = Qbb * dts_tau**3 / 3
        Sig_AB = Qab * (another_leading_2nd_order_decay + dts_tau**2 / 2)
        # # manual Taylor expansion for small dts/tau - Sig_AB
        TE_COEF = [0, 0, 0, 1 / 3, -1 / 8]
        Sig_AB[small_mask] = Qab * polyval(small_dts_tau, TE_COEF)

        ## Sig matrices scaled, e.g. by tau^2
        Sig_Aa = Qaa * decay_m1_sq / 2
        Sig_Ba = -Qab * another_leading_2nd_order_decay
        # # manual Taylor expansion for small dts/tau - Sig_Ba
        TE_COEF = [0, 0, -1 / 2, 1 / 3]
        Sig_Ba[small_mask] = -Qab * polyval(small_dts_tau, TE_COEF)
        Sig_Ab = Qab * leading_2nd_order_decay
        # # manual Taylor expansion for small dts/tau - Sig_Ab
        TE_COEF = [0, 0, 1 / 2, -1 / 6]
        Sig_Ab[small_mask] = Qab * polyval(small_dts_tau, TE_COEF)
        Sig_Bb = Qbb * dts_tau**2 / 2

        ## Sig matrices scaled, e.g. by tau
        Sig_aa = -Qaa * decay_sq_m1 / 2
        Sig_ab = -Qab * decay_m1
        Sig_bb = Qbb * dts_tau

        Q = np.empty((len(self.dts_tau), 4, 4))
        # tau**3 terms
        Q[:, 0, 0] = (taus**2 * Sig_AA + 2 * taus * Sig_AB + Sig_BB) * tau**3
        Q[:, 0, 1] = (-taus * tauc * Sig_AA + (taus - tauc) * Sig_AB + Sig_BB) * tau**3
        Q[:, 1, 1] = (tauc**2 * Sig_AA - 2 * tauc * Sig_AB + Sig_BB) * tau**3
        Q[:, 1, 0] = Q[:, 0, 1].copy()
        # tau**2 terms
        Q[:, 0, 2] = (
            taus**2 * Sig_Aa + taus * Sig_Ba + taus * Sig_Ab + Sig_Bb
        ) * tau**2
        Q[:, 0, 3] = (
            -taus * tauc * Sig_Aa - tauc * Sig_Ba + taus * Sig_Ab + Sig_Bb
        ) * tau**2
        Q[:, 1, 2] = (
            -taus * tauc * Sig_Aa + taus * Sig_Ba - tauc * Sig_Ab + Sig_Bb
        ) * tau**2
        Q[:, 1, 3] = (
            tauc**2 * Sig_Aa - tauc * Sig_Ba - tauc * Sig_Ab + Sig_Bb
        ) * tau**2
        Q[:, 2, 0] = Q[:, 0, 2].copy()
        Q[:, 2, 1] = Q[:, 1, 2].copy()
        Q[:, 3, 0] = Q[:, 0, 3].copy()
        Q[:, 3, 1] = Q[:, 1, 3].copy()
        # tau terms
        Q[:, 2, 2] = (taus**2 * Sig_aa + 2 * taus * Sig_ab + Sig_bb) * tau
        Q[:, 2, 3] = (
            -tauc * taus * Sig_aa - tauc * Sig_ab + taus * Sig_ab + Sig_bb
        ) * tau
        Q[:, 3, 3] = (tauc**2 * Sig_aa - 2 * tauc * Sig_ab + Sig_bb) * tau
        Q[:, 3, 2] = Q[:, 2, 3].copy()

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
        leading_2nd_order_decay[small_mask] = polyval(small_dts_tau, TE_COEF)
        assert (leading_2nd_order_decay >= 0).all(), (
            f"exp(-dts/tau) - 1 + dts/tau cannot be negative: {leading_2nd_order_decay}"
        )

        # Initialize F with direct assignments
        F = np.empty((len(dts_tau), 4, 4), dtype=self.dtype)
        F[:, 0, 0] = 1
        F[:, 0, 1] = 0
        F[:, 0, 2] = tau**2 * (-decay_m1 / tauc + dts_tau / taus)
        F[:, 0, 3] = tau**2 * leading_2nd_order_decay / tauc

        F[:, 1, 0] = 0
        F[:, 1, 1] = 1
        F[:, 1, 2] = tau**2 * leading_2nd_order_decay / taus
        F[:, 1, 3] = tau**2 * (-decay_m1 / taus + dts_tau / tauc)

        F[:, 2, 0] = 0
        F[:, 2, 1] = 0
        F[:, 2, 2] = (tauc + taus * decay) / (tauc + taus)
        F[:, 2, 3] = -tau * decay_m1 / tauc

        F[:, 3, 0] = 0
        F[:, 3, 1] = 0
        F[:, 3, 2] = -tau * decay_m1 / taus
        F[:, 3, 3] = (taus + tauc * decay) / (tauc + taus)

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
        leading_2nd_order_decay[small_mask] = polyval(small_dts_tau, TE_COEF)
        assert (leading_2nd_order_decay >= 0).all(), (
            f"exp(-dts/tau) - 1 + dts/tau cannot be negative: {leading_2nd_order_decay}"
        )
        Na_tauSq_ = Na * tau**2 * leading_2nd_order_decay

        B = np.zeros((len(dts_tau), 4))
        B[:, 0] = taus * Na_tauSq_ + Nb_dtsSq_half
        B[:, 1] = -tauc * Na_tauSq_ + Nb_dtsSq_half
        B[:, 2] = taus * Na_tau_1m_dts_over_tau_decay + Nb_dts
        B[:, 3] = -tauc * Na_tau_1m_dts_over_tau_decay + Nb_dts
        return B

    def _construct_meascov_R(self, R_orig, EFAC, EQUAD):
        return R_orig * EFAC + EQUAD
