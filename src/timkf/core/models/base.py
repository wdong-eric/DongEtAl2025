from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, ClassVar
from timkf.constants import MP_DPS


class ModelConfig(BaseModel):
    """Base configuration model with validation"""

    AVAILABLE_NUMERIC_IMPLS: ClassVar[list[str]] = ["mpmath", "numpy", "jax"]
    numeric_impl: str = Field("mpmath", frozen=True)

    precision: Optional[int] = Field(default=MP_DPS, ge=15, le=100)
    """mpmath options"""

    subtract_linear_trend: bool = Field(default=True)
    """numpy options"""
    Omega_ref: Optional[float] = Field(default=None)
    subtract_quadratic_trend: bool = Field(default=True)
    Omega_dot_ref: Optional[float] = Field(default=None)
    # whether the data is the residuals
    IS_RESIDUAL: bool = Field(default=False)

    @field_validator("numeric_impl")
    @classmethod
    def check_numeric_impl(cls, v):
        if v not in cls.AVAILABLE_NUMERIC_IMPLS:
            raise ValueError("numeric_impl must be either 'mpmath' or 'numpy'")
        return v

    @model_validator(mode="after")
    def check_omega_ref_omega_dot_ref(self):
        if self.numeric_impl == "mpmath":
            return self

        if self.IS_RESIDUAL:
            print("Data is residualised. Checking if used parameters are given.")
            if self.Omega_ref is None or self.Omega_dot_ref is None:
                raise ValueError(
                    "Both Omega_ref and Omega_dot_ref must be provided if IS_RESIDUAL is True"
                )
        if self.subtract_linear_trend:
            if self.Omega_ref is None:
                raise ValueError(
                    "Omega_ref must be provided if subtract_linear_trend is True"
                )
        if self.subtract_quadratic_trend:
            if self.Omega_ref is None or self.Omega_dot_ref is None:
                raise ValueError(
                    "Omega_ref and Omega_dot_ref must be provided if subtract_quadratic_trend is True"
                )
        return self


class MyBaseModel(ABC):
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.numeric_impl = model_config.numeric_impl

    # @abstractmethod
    # def make_fake_data(
    #     self,
    #     times,
    #     tauc,
    #     taus,
    #     Qc,
    #     Qs,
    #     Nc,
    #     Ns,
    #     phic_0,
    #     phis_0,
    #     omgc_0,
    #     omgs_0,
    #     Rc,
    #     seed=917,
    # ):
    #     pass

    @abstractmethod
    def param_map(self, params):
        pass

    @abstractmethod
    def initialise_KFmatricesFQB(self, params, dts):
        """
        Initialise Kalman matrices: F, Q, B.
        """
        pass

    @abstractmethod
    def initial_state_guess(self, params, measurement_R, dt_av):
        """
        Initial states and covariances guess for Kalman filter input: x0, P0.
        """
        pass

    @abstractmethod
    def _construct_transit_F(self, tauc, taus, dts):
        """
        Construct the F matrix for the Kalman filter.
        """
        pass

    @abstractmethod
    def _construct_proccov_Q(self, tauc, taus, Qc, Qs, dts):
        """
        Construct the Q matrix for the Kalman filter.
        """
        pass

    @abstractmethod
    def _construct_control_B(self, tauc, taus, Nc, Ns, dts):
        """
        Construct the B matrix for the Kalman filter.
        """
        pass

    @abstractmethod
    def _construct_meascov_R(self, Rc, dts):
        """
        Construct the R matrix for the Kalman filter.
        """
        pass
