# constants.py
import numpy as np

# ------------------------------ #
# Precision constants
# ------------------------------ #
MP_DPS = 50  # decimal places for npmath.mp

# ------------------------------ #
# Mathematical constants
# ------------------------------ #
pi = np.pi
e = np.e
log10e = np.log10(np.e, dtype=np.float64)
ln10 = np.log(10.0, dtype=np.float64)

TWO_PI = 2 * np.pi

# ------------------------------ #
# Conversion constants
# ------------------------------ #
MICROSECOND_TO_SECONDS = 1e-6  # 1 microsecond = 1e-6 seconds
DAY_TO_SECONDS = 86400  # 1 day = 86400 seconds
MILLISECOND = 1e-3  # s
B_CRIT = 4.414e13  # G; quantum critical threshold for magnetic field; see e.g., https://academic.oup.com/mnras/article/378/1/159/1154016

# ------------------------------ #
# Plotting constants
# ------------------------------ #
QUANTILE_LEVELS = [0.16, 0.5, 0.84]
