from enum import IntEnum, auto
from dataclasses import dataclass

@dataclass(frozen=True)
class MjJtBodyMap:
    jt: str
    body: str


class Keypoint(IntEnum):
    """
    The integer is the index in the MANO ingress data
    """
    lowerArm = 0
    wrist = 1
    thumbProximal = 2
    thumbMedial = 3
    thumbDistal = 4
    thumbTip = 5
    indexProximal = 6
    indexMedial = 7
    indexDistal = 8
    indexTip = 9
    middleProximal = 10
    middleMedial = 11
    middleDistal = 12
    middleTip = 13
    ringProximal = 14
    ringMedial = 15
    ringDistal = 16
    ringTip = 17
    littleProximal = 18
    littleMedial = 19
    littleDistal = 20
    littleTip = 21


@dataclass(frozen=True)
class Keyvector:
    tail: Keypoint
    head: Keypoint
    scale: float
