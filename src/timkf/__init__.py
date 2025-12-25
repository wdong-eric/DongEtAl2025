import logging
import numpy as np

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)  # see time.strftime() for datefmt
src_logger = logging.getLogger(__name__)
src_logger.setLevel(logging.INFO)

# Check if the numpy version is compatible with longdouble
if hasattr(np, "longdouble"):
    print("Numpy version is compatible with longdouble.")
    print("np.finfo(np.longdouble).eps", np.finfo(np.longdouble).eps)
    NP_LONGDOUBLE_TYPE = np.longdouble
    if np.finfo(np.longdouble).eps == np.finfo(np.float64).eps:
        print("np.longdouble is equivalent to np.float64.")
        print("Downcasting to np.float64.")
        NP_LONGDOUBLE_TYPE = np.float64
else:
    print("Numpy version is not compatible with longdouble.")
    print("Using np.float64 as fallback.")
    print("np.finfo(np.float64).eps", np.finfo(np.float64).eps)
    NP_LONGDOUBLE_TYPE = np.float64
print(f"Using NP_LONGDOUBLE_TYPE: {NP_LONGDOUBLE_TYPE}")
