from dataclasses import dataclass
import numpy as np
from typing import Any, Dict

from numpy.typing import NDArray

@dataclass
class HandCfg:
    GC_TENDONS: Dict[str, Any]
    FINGER_TO_TIP: Dict[str, str]
    FINGER_TO_BASE: Dict[str, str]
    GC_LIMITS_LOWER: NDArray[np.float32]
    GC_LIMITS_UPPER: NDArray[np.float32]
