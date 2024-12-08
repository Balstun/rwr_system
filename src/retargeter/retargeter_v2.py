# dependencies for retargeter
import time
from typing import Any, Dict, Tuple
import mujoco
from numpy.typing import NDArray
import torch
import numpy as np

from src.retargeter.hand_cfgs.hand_cfg import HandCfgV2
from src.retargeter.hand_cfgs.structures import Keyvector
from .utils import retarget_utils
from scipy.spatial.transform import Rotation

class RetargeterV2:
    """
    Retargeter than uses mujoco forward kinematics to figuire out the best joint angles.

    High Level Idea:
        Guess and optimize over actuator angles (in mujoco corresponding to joint angles) and perform forward kinematics using the mujoco model.
    """

    def _load_mujoco_model(self, mjcf_filepath: str):
        self.model = mujoco.MjModel.from_xml_path(mjcf_filepath)
        self.data = mujoco.MjData(self.model)

    def apply_fk(self, xpos: NDArray[np.float64]):
        """
        Given Actuator Angles, find the correct joint angles
        """

        assert(xpos.shape == (self.num_actuators, 1))
        assert(self.hand_config.ACTUATOR2JOINT_MAT.shape == (self.num_joints, self.num_actuators))

        # Convert xpos into qpos
        qpos = self.hand_config.ACTUATOR2JOINT_MAT.dot(xpos)
        self.data.qpos = qpos

        # Perform forward kinematics 
        mujoco.mj_forward(self.model, self.data)

    def _mano_keyvector(self, mano_xpos: torch.Tensor, spec: Keyvector) -> Tuple[torch.Tensor, np.float32]:
        """
        Given the key vector spec and the mano cartesian poses, get the vector from tail to head in the world frame,
        get the vector norm 
        """
        # xpos and xquat are represented in the world frame (W)
        r_W_WTail = mano_xpos[spec.tail]
        r_W_WHead = mano_xpos[spec.head]
        r_W_TailHead = r_W_WHead - r_W_WTail 

        return r_W_TailHead, np.linalg.norm(r_W_TailHead) * spec.scale

    def _mj_keyvector(self, spec: Keyvector) -> Tuple[NDArray[np.float32], NDArray[np.float32], np.float32]:
        """
        Called after performing FK
        Given the keyvector spec, identify the bodies to measure for the keypoint

        Get the xpos of the bodies, convert to the position vector in the WORLD AND TAIL FRAME.

        Returns:
            - vector from tail to head in the world frame
            - vector from tail to head in the tail frame (for visualization)
            - norm of the vector scaled by the provided scale factor
        """
        b_tail = self.model.bodies(self.hand_config.KEYPOINT_MJ_MAPPING[spec.tail].body)
        b_head = self.model.bodies(self.hand_config.KEYPOINT_MJ_MAPPING[spec.head].body)

        # xpos and xquat are represented in the world frame (W)
        r_W_WTail = b_tail.xpos
        r_W_WHead = b_head.xpos
        r_W_TailHead = (r_W_WHead - r_W_WTail ).reshape((3, 1))

        C_W_Tail = b_tail.xmat.reshape((3, 3))

        r_Tail_TailHead = C_W_Tail.T @ r_W_TailHead

        return r_W_TailHead, r_Tail_TailHead, np.linalg.norm(r_Tail_TailHead) * spec.scale

    def __init__(
        self,
        hand_config: HandCfgV2,
        mjcf_filepath: str,
        device: str = "cpu",
    ) -> None:

        self.device = device
        self.hand_config: HandCfgV2 = hand_config
        self.num_active_keyvectors = len(self.hand_config.KEYVECTORS)

        self.loss_coeffs = torch.tensor([5.0] * self.num_active_keyvectors).to(device)

        self._load_mujoco_model(mjcf_filepath)

        # Number of joint actuators that MUJOCO has
        self.num_joints = self.model.nq
        self.num_actuators = self.data.ctrl.shape[0]

        self.retarget_config: Dict[str, Any] = {
            "learning_rate": 2.5
        }

        # Initialize the gc_joints tensor
        self.joint_angles = torch.ones(self.num_actuators).to(self.device)
        self.joint_angles.requires_grad_()

        # Initialize the optimizer 
        # self.opt = torch.optim.Adam([self.gc_joints], lr=self.lr)
        self.opt = torch.optim.RMSprop([self.joint_angles], lr=self.retarget_config["learning_rate"])

        self.root = torch.zeros(1, 3).to(self.device)


    def retarget_finger_mano_joints(
        self,
        mano_xpos: torch.Tensor,
        warm: bool = True,
        opt_steps: int = 2,
        debug_dict=None,
    ):
        """
        Process the MANO joints and update the finger joint angles
        joints: (22, 3)
        """

        print(f"Retargeting: Warm: {warm} Opt steps: {opt_steps}")

        start_time = time.time()
        if not warm:
            # Initial Guess
            self.joint_angles = torch.ones(self.num_actuators).to(self.device) * 15.0
            self.joint_angles.requires_grad_()

        assert mano_xpos.shape == (
            22,
            3,
        ), "The shape of the mano joints array should be (22, 3)"

        # Get keyvectors from mano
        mano_vecs_norm = torch.tensor([self._mano_keyvector(mano_xpos, vec)[1] for vec in self.hand_config.KEYVECTORS])

        for step in range(opt_steps):

            self.apply_fk(self.joint_angles.detach().cpu().numpy())

            # Get Mujoco Vectors
            mj_vecs_norm = []
            mj_vecs_local = [] # For debug purposes
            for vec in self.hand_config.KEYVECTORS:
                _, r_local, norm = self._mj_keyvector(vec)
                mj_vecs_local.append(r_local)
                mj_vecs_norm.append(norm)

            mj_vecs_norm = torch.tensor(mj_vecs_norm)

            loss: torch.Tensor = torch.norm(self.loss_coeffs * (mano_vecs_norm - mj_vecs_norm))

            # loss += torch.sum(
            #     self.regularizer_weights * (self.gc_joints - self.regularizer_zeros) ** 2
            # )

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            with torch.no_grad():
                self.joint_angles[:] = torch.clamp(
                    self.joint_angles,
                    torch.tensor(self.hand_config.GC_LIMITS_LOWER).to(self.device),
                    torch.tensor(self.hand_config.GC_LIMITS_UPPER).to(self.device),
                )
                
        finger_joint_angles = self.joint_angles.detach().cpu().numpy()

        if debug_dict:
            if "keyvec_mujoco" not in debug_dict.keys():
                debug_dict["keyvec_mujoco"] = {}

            debug_dict["keyvec_mujoco"]["start"] = start_vectors_untuned
            debug_dict["keyvec_mujoco"]["end"] = end_vectors_untuned

        print(f"Retarget time: {(time.time() - start_time) * 1000} ms")

        return finger_joint_angles, debug_dict

    def retarget(self, joints, debug_dict=None):
        elbow_marker_active = True
        if elbow_marker_active:
            joints = np.delete(joints, 1, axis=0)

        normalized_joint_pos, mano_center_and_rot = (
            retarget_utils.normalize_points_to_hands_local(joints)
        )
        # normalized_joint_pos = self.adjust_mano_fingers(normalized_joint_pos)
        normalized_joint_pos = (
            normalized_joint_pos @ self.model_rotation.T + self.model_center
        )
        if debug_dict is not None:
            debug_dict["mano_center_and_rot"] = mano_center_and_rot
            debug_dict["model_center_and_rot"] = (self.model_center, self.model_rotation)
            debug_dict["normalized_joint_pos"] = normalized_joint_pos
        self.target_angles, debug_dict = self.retarget_finger_mano_joints(normalized_joint_pos, debug_dict=debug_dict)
        return self.target_angles, debug_dict
