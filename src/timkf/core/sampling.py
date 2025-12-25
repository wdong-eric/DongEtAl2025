import bilby
from .kalman import KalmanFilterStandard


class BaseKFLikelihood(bilby.Likelihood):
    def __init__(self, kalman_filter: KalmanFilterStandard, parameters, ll_burn=0):
        if parameters is None:
            parameters = self.default_parameters()
        super().__init__(parameters=parameters)
        self.KF = kalman_filter
        self.ll_burn = ll_burn

    def default_parameters(self):
        raise NotImplementedError(
            "BaseKFLikelihood does not have default parameters. Please implement the default_parameters() method in a subclass, when parameters=None."
        )

    def log_likelihood(self):
        return self.KF.get_likelihood(self.parameters, self.ll_burn)


class TwoCompFreqKFLikelihood(BaseKFLikelihood):
    def default_parameters(self):
        return {
            "ratio": None,
            "tau": None,
            "Qc": None,
            "Qs": None,
            "lag": None,
            "omgc_dot": None,
            "omgc_0": None,
            "omgs_0": None,
            "EFAC": None,
            "EQUAD": None,
        }


class TwoCompPhaseKFLikelihood(BaseKFLikelihood):
    def default_parameters(self):
        return {
            "ratio": None,
            "tau": None,
            "Qc": None,
            "Qs": None,
            "lag": None,
            "omgc_dot": None,
            "omgc_0": None,
            "omgs_0": None,
            "phic_0": None,
            "phis_0": None,
            "EFAC": None,
            "EQUAD": None,
        }


class OneCompWTNPhaseKFLikelihood(BaseKFLikelihood):
    """
    The likelihood for the Kalman filter with White Timing Noise hypothesis, i.e. only fitting for deterministic pulsar paramters, EFAC and EQUAD only.
    """

    def default_parameters(self):
        return {
            "omgc_dot": None,
            "omgc_0": None,
            "phic_0": None,
            "EFAC": None,
            "EQUAD": None,
        }
