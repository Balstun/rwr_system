from typing import Dict, List
import numpy as np
from numpy.typing import NDArray

from retargeter.hand_cfgs.structures import Keypoint, Keyvector, MjJtBodyMap
from retargeter.hand_cfgs.hand_cfg import HandCfgV2

"""
Note that the ordering of the keypoints is EXACTLY the same as the ingress data
"""
KEYPOINT_MJ_MAPPING: Dict[Keypoint, MjJtBodyMap] = {
    Keypoint.lowerArm: MjJtBodyMap("", ""), # no mujoco mapping
    Keypoint.wrist: MjJtBodyMap("root2palm", "root"),
    Keypoint.thumbProximal: MjJtBodyMap("root2thumb_base", "thumb_base"),
    Keypoint.thumbMedial: MjJtBodyMap("thumb_base2pp", "thumb_mp"),
    Keypoint.thumbDistal: MjJtBodyMap("thumb_pp2mp", "thumb_dp"),
    Keypoint.thumbTip: MjJtBodyMap("thumb_mp2dp", "thumb_fingertip"),
    Keypoint.indexProximal: MjJtBodyMap("index_base2adb", "index_base"),
    Keypoint.indexMedial: MjJtBodyMap("index_adb2pp", "index_pp_link"),
    Keypoint.indexDistal: MjJtBodyMap("index_pp2mp", "index_mp_link"),
    Keypoint.indexTip: MjJtBodyMap("index_mp2dp", "index_fingertip"),
    Keypoint.middleProximal: MjJtBodyMap("middle_base2adb", "middle_base"),
    Keypoint.middleMedial: MjJtBodyMap("middle_adb2pp", "middle_pp_link"),
    Keypoint.middleDistal: MjJtBodyMap("middle_pp2mp", "middle_mp_link"),
    Keypoint.middleTip: MjJtBodyMap("middle_mp2dp", "middle_fingertip"),
    Keypoint.ringProximal: MjJtBodyMap("ring_base2adb", "ring_base"),
    Keypoint.ringMedial: MjJtBodyMap("ring_adb2pp", "ring_pp_link"),
    Keypoint.ringDistal: MjJtBodyMap("ring_pp2mp", "ring_mp_link"),
    Keypoint.ringTip: MjJtBodyMap("ring_mp2dp", "ring_fingertip"),
    Keypoint.littleProximal: MjJtBodyMap("pinky_base2adb", "pinky_base"),
    Keypoint.littleMedial: MjJtBodyMap("pinky_adb2pp", "pinky_pp_link"),
    Keypoint.littleDistal: MjJtBodyMap("pinky_pp2mp", "pinky_mp_link"),
    Keypoint.littleTip: MjJtBodyMap("pinky_mp2dp", "pinky_fingertip")
}

GC_LIMITS_LOWER = np.array(
    [
        0.0,  # root2thumb_base
        -45.0,  # thumb_base2pp
        0.0,  # thumb_pp2mp_virt
        -30.0,  # index_base2abd_virt
        0.0,  # index_adb2pp_virt
        0.0,  # index_pp2mp_virt
        -30.0,  # middle_base2abd_virt
        0.0,  # middle_adb2pp_virt
        0.0,  # middle_pp2mp_virt
        -30.0,  # ring_base2abd_virt
        0.0,  # ring_adb2pp_virt
        0.0,  # ring_pp2mp_virt
        -30.0,  # pinky_base2abd_virt
        0.0,  # pinky_adb2pp_virt
        0.0,  # pinky_pp2mp_virt
    ]
)
GC_LIMITS_UPPER = np.array(
    [
        90.0,  # root2thumb_base
        45.0,  # thumb_base2pp
        90.0,  # thumb_pp2mp_virt
        30.0,  # index_base2adb_virt
        90.0,  # index_adb2pp_virt
        90.0,  # index_pp2mp_virt
        30.0,  # middle_base2abd_virt
        90.0,  # middle_adb2pp_virt
        90.0,  # middle_pp2mp_virt
        30.0,  # ring_base2adb_virt
        90.0,  # ring_adb2pp_virt
        90.0,  # ring_pp2mp_virt
        30.0,  # pinky_base2adb_virt
        90.0,  # pinky_adb2pp_virt
        90.0,  # pinky_pp2mp_virt
    ]
)

def actuator2jointMatThumb() -> NDArray[np.float32]:
    # For the finger, each joint maps to an actuator angle linearly based on the tendon and gearing constraints
    # [ root2thumb_base   ]
    # [ base2pp        ]
    # [ pp2mp_virt      ]
    # [ pp2mp           ]
    # [ mp2dp_virt      ]
    # [ mp2dp           ]

    # Actuator Order
    # [ palm ] - 1 for root2thumb_base
    # [ adb ] -  1 for base2pp
    # [ mcp ] - 0.5 for pp2mp, 0.5*0.0.71 for mp2dp
    pinJointRatio = 1
    rollingContactJointRatio = 0.5
    passiveDpRatio = 0.71

    return np.array([
        [pinJointRatio, 0, 0],
        [0, pinJointRatio, 0],
        [0, 0, rollingContactJointRatio],
        [0, 0, rollingContactJointRatio],
        [0, 0, rollingContactJointRatio*passiveDpRatio],
        [0, 0, rollingContactJointRatio*passiveDpRatio],
    ], dtype=np.float32)

def actuator2jointMatWrist() -> NDArray[np.float32]:
    return np.ones(1, dtype=np.float32)

def actuator2jointMatFinger() -> NDArray[np.float32]:
    # For the finger, each joint maps to an actuator angle linearly based on the tendon and gearing constraints
    # [ base2adb_virt   ]
    # [ base2adb        ]
    # [ adb2pp_virt     ]
    # [ adb2pp          ]
    # [ pp2mp_virt      ]
    # [ pp2mp           ]
    # [ mp2dp_virt      ]
    # [ mp2dp           ]

    # Actuator Order
    # [ mcp ] - 0.5 for the adb2pp angles
    # [ pip ] - 0.5 for pp2mp and 0.5*0.71 for mp2dp
    # [ adb ] - 0.25 for base2adbp
    rollingContactJointRatio = 0.5
    adbGearing = 0.5
    passiveDpRatio = 0.71

    return np.array([
        [0, 0, rollingContactJointRatio*adbGearing],
        [0, 0, rollingContactJointRatio*adbGearing],
        [rollingContactJointRatio, 0, 0],
        [rollingContactJointRatio, 0, 0],
        [0, rollingContactJointRatio, 0],
        [0, rollingContactJointRatio*passiveDpRatio, 0],
        [0, rollingContactJointRatio*passiveDpRatio, 0],
        [0, rollingContactJointRatio*passiveDpRatio, 0],
    ], dtype=np.float32)

ACTUATOR2JOINT_MAT: NDArray[np.float32] = np.vstack((actuator2jointMatWrist(),) + (actuator2jointMatFinger(), )*4 + (actuator2jointMatThumb(),))

KEYVECTORS: List[Keyvector] = [
    Keyvector(Keypoint.wrist, Keypoint.thumbTip, 1.0)
]

BiomimicHandCfg = HandCfgV2(KEYVECTORS, KEYPOINT_MJ_MAPPING, ACTUATOR2JOINT_MAT, GC_LIMITS_LOWER, GC_LIMITS_UPPER)
