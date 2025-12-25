from .hierarchical_lib import *
from .hyper_model import *

HYPER_MODEL_MAPPING_DICT = {
    "L10N": Log10normHyperModel,
    # "L10N_L10U": Log10normLog10unifHyperModel,
    # "L10N_L10UFB": Log10normLog10unifFixedBdryHyperModel,
    "L10N_L10UF1B": Log10normLog10unifFixedOneBdryHyperModel,
    # "L10N_L10U_PCat": Log10normLog10unifPreCategorisedHyperModel,
    # "L10N_L10UFB_PCat": Log10normLog10unifFixedBdryPreCategorisedHyperModel,
    "L10N_L10UF1B_PCat": Log10normLog10unifFixedOneBdryPreCategorisedHyperModel,
    "L10N_L10N": TwoLog10normHyperModel,
    # "L10N_L10N_L10UF1B": TwoLog10normLog10unifFixedOneBdryHyperModel,
    "PL_CorrAgnst_Omgc_Omgcdot": PowerLawCorrAgainstOmgcOmgcdotHyperModel,
    "PL_CorrAgnst_Omgc_Omgcdot_Tobs": PowerLawCorrAgainstOmgcOmgcdotTobsHyperModel,
}
