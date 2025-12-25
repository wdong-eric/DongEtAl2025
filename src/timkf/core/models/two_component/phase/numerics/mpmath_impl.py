"""mpmath implementation of two-component phase"""

import numpy as np
from mpmath import mp
from ....base import ModelConfig, MyBaseModel


class MPMathTwoComponentPhase(MyBaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self._configure_mpmath()

    def _configure_mpmath(self):
        mp.dps = self.config.precision
        self.mp_array_exp = np.frompyfunc(mp.exp, 1, 1)

    @staticmethod
    def param_map(params):
        TAU, LAG, OMGC_DOT = (
            params["tau"],
            params["lag"],
            params["omgc_dot"],
        )
        if "ratio" in params:
            RATIO = params["ratio"]
        else:
            RATIO = 1 / params["xs"] - 1
        taus = (1 + RATIO) * TAU
        tauc = (1 + RATIO) / RATIO * TAU
        Nc = OMGC_DOT + RATIO / (1 + RATIO) * LAG / TAU
        Ns = OMGC_DOT - LAG / TAU * (1 + RATIO) ** -1
        Qc = params["Qc"]
        if "tauprime" in params:
            TAUPRIME = params["tauprime"]
            QSC = ((1 / TAUPRIME**2) - (1 / taus**2)) * tauc**2
            Qs = QSC * Qc
        else:
            Qs = params["Qs"]

        tauc = tauc * mp.mpf(1)
        taus = taus * mp.mpf(1)
        Qc = Qc * mp.mpf(1)
        Qs = Qs * mp.mpf(1)
        Nc = Nc * mp.mpf(1)
        Ns = Ns * mp.mpf(1)
        return tauc, taus, Qc, Qs, Nc, Ns

    def initialise_KFmatricesFQB(self, params, dts):
        """
        Initialise Kalman matrices: F, Q, B.
        """
        tauc, taus, Qc, Qs, Nc, Ns = self.param_map(params)

        F = self._construct_transit_F(tauc, taus, dts)
        Q = self._construct_proccov_Q(tauc, taus, Qc, Qs, dts)
        B = self._construct_control_B(tauc, taus, Nc, Ns, dts)

        return F, Q, B

    def initial_state_guess(self, params: dict, measurement_R, dt_av):
        """
        Initial states and covariances guess for Kalman filter input: x0, P0.
        """
        tauc, taus, _, _, Nc, Ns = self.param_map(params)
        lag = (tauc * taus / (tauc + taus)) * (Nc - Ns)
        phic_0 = params.get("phic_0") * mp.mpf(1)
        phis_0 = params.get("phis_0") * mp.mpf(1)
        omgc_0 = params["omgc_0"] * mp.mpf(1)
        omgs_0 = (
            omgc_0 - lag if params.get("omgs_0", None) is None else params["omgs_0"]
        )

        phase_cov = measurement_R[0]
        freq_cov = phase_cov / (dt_av) ** 2

        X0 = mp.matrix([phic_0, phis_0, omgc_0, omgs_0])
        P0 = mp.diag([phase_cov, phase_cov, freq_cov, freq_cov])

        return X0, P0

    def _construct_proccov_Q(self, tauc, taus, Qc, Qs, dts):
        tau = 1 / (1 / tauc + 1 / taus)
        Qaa = (Qc + Qs) / (tauc + taus) ** 2
        Qab = (Qc * tauc - Qs * taus) / (tauc + taus) ** 2
        Qbb = (Qc * tauc**2 + Qs * taus**2) / (tauc + taus) ** 2
        dts = dts * mp.mpf(1)

        # High precision exponential version (mpmath)
        Sig_AA = (
            Qaa
            * (
                dts / tau
                - (1 - self.mp_array_exp(-dts / tau))
                - (1 / 2) * (1 - self.mp_array_exp(-dts / tau)) ** 2
            )
            * tau**3
        )
        Sig_BB = Qbb * dts**3 / 3
        # it seems like even mpmath would be affected by the operation order
        Sig_AB = (
            Qab
            * (
                (1 + dts / tau) * self.mp_array_exp(-dts / tau)
                - (1 - dts**2 / (2 * tau**2))
            )
            # * (
            #     self.mp_array_exp(-dts / tau)
            #     - 1
            #     + dts / tau * self.mp_array_exp(-dts / tau)
            #     + dts**2 / (2 * tau**2)
            # )
            * tau**3
        )
        Sig_Aa = Qaa * (tau**2 / 2) * (1 - self.mp_array_exp(-dts / tau)) ** 2
        Sig_Ba = Qab * tau**2 * (1 - (1 + dts / tau) * self.mp_array_exp(-dts / tau))
        Sig_aa = Qaa * (tau / 2) * (1 - self.mp_array_exp(-2 * dts / tau))
        Sig_Ab = Qab * tau**2 * (self.mp_array_exp(-dts / tau) - (1 - dts / tau))
        Sig_Bb = Qbb * dts**2 / 2
        Sig_ab = Qab * tau * (1 - self.mp_array_exp(-dts / tau))
        Sig_bb = Qbb * dts

        Q = np.zeros((len(dts), 4, 4), dtype=mp.mpf)
        Q[:, 0, 0] = taus**2 * Sig_AA + 2 * taus * Sig_AB + Sig_BB
        Q[:, 0, 1] = -taus * tauc * Sig_AA + (taus - tauc) * Sig_AB + Sig_BB
        Q[:, 0, 2] = taus**2 * Sig_Aa + taus * Sig_Ba + taus * Sig_Ab + Sig_Bb
        Q[:, 0, 3] = -taus * tauc * Sig_Aa - tauc * Sig_Ba + taus * Sig_Ab + Sig_Bb

        Q[:, 1, 0] = Q[:, 0, 1].copy()
        Q[:, 1, 1] = tauc**2 * Sig_AA - 2 * tauc * Sig_AB + Sig_BB
        Q[:, 1, 2] = -taus * tauc * Sig_Aa + taus * Sig_Ba - tauc * Sig_Ab + Sig_Bb
        Q[:, 1, 3] = tauc**2 * Sig_Aa - tauc * Sig_Ba - tauc * Sig_Ab + Sig_Bb

        Q[:, 2, 0] = Q[:, 0, 2].copy()
        Q[:, 2, 1] = Q[:, 1, 2].copy()
        Q[:, 2, 2] = taus**2 * Sig_aa + 2 * taus * Sig_ab + Sig_bb
        Q[:, 2, 3] = -tauc * taus * Sig_aa - tauc * Sig_ab + taus * Sig_ab + Sig_bb

        Q[:, 3, 0] = Q[:, 0, 3].copy()
        Q[:, 3, 1] = Q[:, 1, 3].copy()
        Q[:, 3, 2] = Q[:, 2, 3].copy()
        Q[:, 3, 3] = tauc**2 * Sig_aa - 2 * tauc * Sig_ab + Sig_bb

        return Q

    def _construct_transit_F(self, tauc, taus, dts):
        tau = 1 / (1 / tauc + 1 / taus)
        F = np.zeros((len(dts), 4, 4), dtype=mp.mpf)
        F[:, 0, 0] = np.ones(len(dts))
        F[:, 0, 1] = np.zeros(len(dts))
        F[:, 0, 2] = (tau * taus * (1 - self.mp_array_exp(-dts / tau)) + tauc * dts) / (
            tauc + taus
        )
        F[:, 0, 3] = (tau * taus * (self.mp_array_exp(-dts / tau) - 1 + dts / tau)) / (
            tauc + taus
        )

        F[:, 1, 0] = np.zeros(len(dts))
        F[:, 1, 1] = np.ones(len(dts))
        F[:, 1, 2] = (tau * tauc * (self.mp_array_exp(-dts / tau) - 1 + dts / tau)) / (
            tauc + taus
        )
        F[:, 1, 3] = (tau * tauc * (1 - self.mp_array_exp(-dts / tau)) + taus * dts) / (
            tauc + taus
        )

        F[:, 2, 0] = np.zeros(len(dts))
        F[:, 2, 1] = np.zeros(len(dts))
        F[:, 2, 2] = (tauc + taus * self.mp_array_exp(-dts / tau)) / (tauc + taus)
        F[:, 2, 3] = (taus * (1 - self.mp_array_exp(-dts / tau))) / (tauc + taus)

        F[:, 3, 0] = np.zeros(len(dts))
        F[:, 3, 1] = np.zeros(len(dts))
        F[:, 3, 2] = (tauc * (1 - self.mp_array_exp(-dts / tau))) / (tauc + taus)
        F[:, 3, 3] = (taus + tauc * self.mp_array_exp(-dts / tau)) / (tauc + taus)

        return F

    def _construct_control_B(self, tauc, taus, Nc, Ns, dts):
        tau = (tauc * taus) / (tauc + taus)
        Na = (Nc - Ns) / (tauc + taus)
        Nb = (tauc * Nc + taus * Ns) / (tauc + taus)

        B = np.zeros((len(dts), 4), dtype=mp.mpf)
        B[:, 0] = taus * (
            Na * tau**2 * (self.mp_array_exp(-dts / tau) - (1 - dts / tau))
        ) + (Nb * dts**2 / 2)
        B[:, 1] = -tauc * (
            Na * tau**2 * (self.mp_array_exp(-dts / tau) - (1 - dts / tau))
        ) + (Nb * dts**2 / 2)
        B[:, 2] = taus * (Na * tau * (1 - self.mp_array_exp(-dts / tau))) + (Nb * dts)
        B[:, 3] = -tauc * (Na * tau * (1 - self.mp_array_exp(-dts / tau))) + (Nb * dts)
        return B

    def _construct_meascov_R(self, R_orig, EFAC, EQUAD):
        return R_orig * EFAC + EQUAD