from typing import Dict, Tuple

import torch
import numpy as np


def get_mano_joints_dict(
    joints: torch.Tensor, include_wrist=False, batch_processing=False
):
    # joints: 21 x 3
    # For retargeting, we don't need the wrist
    # For visualization, we need the wrist
    if not batch_processing:
        if not include_wrist:
            return {
                "thumb": joints[1:5, :],
                "index": joints[5:9, :],
                "middle": joints[9:13, :],
                "ring": joints[13:17, :],
                "pinky": joints[17:21, :],
            }
        else:
            return {
                "wrist": joints[0, :],
                "thumb": joints[1:5, :],
                "index": joints[5:9, :],
                "middle": joints[9:13, :],
                "ring": joints[13:17, :],
                "pinky": joints[17:21, :],
            }
    else:
        if not include_wrist:
            return {
                "thumb": joints[:, 1:5, :],
                "index": joints[:, 5:9, :],
                "middle": joints[:, 9:13, :],
                "ring": joints[:, 13:17, :],
                "pinky": joints[:, 17:21, :],
            }
        else:
            return {
                "wrist": joints[:, 0, :],
                "thumb": joints[:, 1:5, :],
                "index": joints[:, 5:9, :],
                "middle": joints[:, 9:13, :],
                "ring": joints[:, 13:17, :],
                "pinky": joints[:, 17:21, :],
            }


def get_mano_fingertips_batch(mano_joints_dict):
    return {
        "thumb": mano_joints_dict["thumb"][:, [3], :],
        "index": mano_joints_dict["index"][:, [3], :],
        "middle": mano_joints_dict["middle"][:, [3], :],
        "ring": mano_joints_dict["ring"][:, [3], :],
        "pinky": mano_joints_dict["pinky"][:, [3], :],
    }


def get_mano_pps_batch(mano_joints_dict):
    return {
        "thumb": mano_joints_dict["thumb"][:, [0], :],
        "index": mano_joints_dict["index"][:, [0], :],
        "middle": mano_joints_dict["middle"][:, [0], :],
        "ring": mano_joints_dict["ring"][:, [0], :],
        "pinky": mano_joints_dict["pinky"][:, [0], :],
    }


def get_keyvectors(finger_bases: Dict[str, torch.Tensor], fingertips: Dict[str, torch.Tensor], other_pts: Dict[str, torch.Tensor], apply_scaling=True):

    keyvectors_data: Dict[str, Tuple[torch.Tensor, torch.Tensor, float]] = {
        # vector_name: (source, end, scaling)
        # Define keyvectors for the palm to each fingertip
        "wrist2thumb": (other_pts["wrist"], fingertips["thumb"], 0.2),
        "wrist2index": (other_pts["wrist"], fingertips["index"], 0.5),
        "wrist2middle": (other_pts["wrist"], fingertips["middle"], 0.5),
        "wrist2ring": (other_pts["wrist"], fingertips["ring"], 0.5),
        "wrist2pinky": (other_pts["wrist"], fingertips["pinky"], 0.5),
        # Define keyvectors from finger bases to fingertips
        # "thumbBase2thumbTip": (finger_bases["thumb"], fingertips["thumb"], 0.2),
        # "indexBase2indexTip": (finger_bases["index"], fingertips["index"], 1.0),
        # "middleBase2middleTip": (finger_bases["middle"], fingertips["middle"], 1.0),
        # "ringBase2ringTip": (finger_bases["ring"], fingertips["ring"], 1.0),
        # "pinkyBase2pinkyTip": (finger_bases["pinky"], fingertips["pinky"], 1.0),
        # Define keyvectors for all adjacent fingers
        "thumbTip2indexTip": (fingertips["thumb"], fingertips["index"], 0.2),
        "indexTip2middleTip": (fingertips["index"], fingertips["middle"], 0.2),
        "middleTip2ringTip": (fingertips["middle"], fingertips["ring"], 0.2),
        "ringTip2pinkyTip": (fingertips["ring"], fingertips["pinky"], 0.2),
        # Define keyvectors for each fingertip to the thumb fingertip
        "thumb2middle": (fingertips["thumb"], fingertips["middle"], 0.5),
        "thumb2ring": (fingertips["thumb"], fingertips["ring"], 0.5),
        "thumb2pinky": (fingertips["thumb"], fingertips["pinky"], 0.5),
        # Define additional keyvector for thumb
        "pinky2thumb": (finger_bases["pinky"], fingertips["thumb"], 0.5),
        "thumbKnuckle2thumbTip": (other_pts["thumb_knuckle"], fingertips["thumb"], 0.3),
        # Aduction/Abduction keyvectors
        "index2middle_adb": (other_pts["index_knuckle"], other_pts["middle_knuckle"], 0.5),
        "middle2ring_adb": (other_pts["middle_knuckle"], other_pts["ring_knuckle"], 0.5),
        "ring2pinky_adb": (other_pts["ring_knuckle"], other_pts["pinky_knuckle"], 0.5),
        # Testing/Debugging keyvectors
        # "indexBase2indexTip": (finger_bases["index"], fingertips["index"], 0.5),
        # "indexBase2indexKnuckle": (finger_bases["index"], other_pts["index_knuckle"], 0.5),
        # "indexKnuckle2indexTip": (other_pts["index_knuckle"], fingertips["index"], 0.5),
    }
    
    keyvectors = {}
    for key, value in keyvectors_data.items():
        start, end, scale = value
        keyvectors[key] = (end - start) * (scale if apply_scaling else 1)
    return keyvectors_data, keyvectors

def rotation_matrix_z(angle):
    """
    Returns a 3x3 rotation matrix about the z-axis for the given angle.
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rot_mat = np.array(
        [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
    )
    return rot_mat


def rotation_matrix_y(angle):
    """
    Returns a 3x3 rotation matrix about the y-axis for the given angle.
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rot_mat = np.array(
        [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
    )
    return rot_mat


def rotation_matrix_x(angle):
    """
    Returns a 3x3 rotation matrix about the x-axis for the given angle.
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rot_mat = np.array(
        [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]]
    )
    return rot_mat


def get_hand_center_and_rotation(
    thumb_base, index_base, middle_base, ring_base, pinky_base, wrist=None
):
    """
    Get the center of the hand and the rotation matrix of the hand
    x axis is the direction from ring to index finger base
    y axis is the direction from wrist to middle finger base
    z axis goes from the palm if the hand is right hand, otherwise it goes to the palm
    If the hand is right hand, then the z
    """
    hand_center = (thumb_base + pinky_base) / 2
    hand_center = hand_center
    if wrist is None:
        wrist = hand_center

    y_axis = middle_base - wrist
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = index_base - ring_base
    x_axis -= (x_axis @ y_axis.T) * y_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    rot_matrix = np.concatenate(
        (x_axis.reshape(1, 3), y_axis.reshape(1, 3), z_axis.reshape(1, 3)), axis=0
    ).T
    return hand_center, rot_matrix


def normalize_points_to_hands_local(joint_pos):
    """
    param: joint_pos, a numpy array of 3D joint positions (MANO format)

    Returns the joint positions with normalized translation and rotation.
    """

    # construct a plane from wrist, first index finger joint, first pinky joint
    joint_dict = get_mano_joints_dict(joint_pos, include_wrist=True)
    hand_center, hand_rot = get_hand_center_and_rotation(
        thumb_base=joint_dict["thumb"][0],
        index_base=joint_dict["index"][0],
        middle_base=joint_dict["middle"][0],
        ring_base=joint_dict["ring"][0],
        pinky_base=joint_dict["pinky"][0],
        wrist=joint_dict["wrist"],
    )
    joint_pos = joint_pos - hand_center
    joint_pos = joint_pos @ hand_rot
    return joint_pos, (hand_center, hand_rot)


def get_unoccluded_hand_joint_idx(joint_pos):
    """
    param: joint_pos, a numpy array of 3D joint positions (MANO format), not normalized
    Returns the joint that has the least z value and should be visible in the image (y value is in the direction of the camera).
    We can then project this joint into 3D space, and then from there get the 3D position of the wrist (which may be occluded)
    """

    # get the joint with the lowest z value (closest to camera)
    max_joint_idx = np.argmin(joint_pos[:, 2])
    return max_joint_idx


def get_wrist_translation(joint_idx, joint_pos):
    """
    param: joint_idx, the index of the joint with the highest y value
    param: joint_pos, a numpy array of 3D joint positions (MANO format), not normalized
    Returns the translation of the wrist in the hand frame relative to the joint_idx joint
    """

    # get the 3D position of the wrist
    joint = joint_pos[joint_idx, :]
    wrist = joint_pos[0, :]

    return wrist - joint


def rolling_average_filter(positions, new_pos):
    """
    A rolling average filter for the wrist position.
    param: positions, a numpy array of 3D positions of the wrist
    param: new_pos, a numpy array of the new 3D position of the wrist
    """

    positions = np.roll(positions, -1, axis=0)
    positions[-1, :] = new_pos

    return positions, np.nanmean(positions, axis=0)


# Actually not used in frankmocap default
def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from
    https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.reshape(-1, 3, 1)
    y = y.reshape(-1, 3, 1)
    z = z.reshape(-1, 3, 1)
    matrix = np.concatenate((x, y, z), 2)  # batch*3*3
    return matrix


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = np.sqrt((v**2).sum(1))  # batch
    v_mag = np.maximum(v_mag, np.array([1e-8]))
    v_mag = np.broadcast_to(v_mag.reshape(batch, 1), (batch, v.shape[1]))
    v = v / v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = np.concatenate(
        (i.reshape(batch, 1), j.reshape(batch, 1), k.reshape(batch, 1)), 1
    )

    return out


def normalize_points_rokoko(
    joint_pos, mirror_x=False, flip_y_axis=False, flip_x_axis=False
):
    # normalize the translation of the hand: set the wrist point to zero
    wrist_point = joint_pos[0, :]
    joint_pos -= wrist_point

    # construct a plane from wrist, first index finger joint, first pinky joint
    joint_dict = get_mano_joints_dict(joint_pos, include_wrist=True)
    wrist_point = joint_dict["wrist"]
    middle_point = joint_dict["middle"][0]
    pinky_point = joint_dict["pinky"][0]
    # find basis vectors for the plane
    base_1 = middle_point - wrist_point
    base_2 = pinky_point - wrist_point
    normal_vec = np.cross(base_1, base_2)
    base_2 = np.cross(normal_vec, base_1)

    # normalize basis vectors
    normal_vec /= np.linalg.norm(normal_vec)
    base_1 /= np.linalg.norm(base_1)
    base_2 /= np.linalg.norm(base_2)

    # construct the matrix for the base change from the hand frame basis vectors
    base_matrix = np.zeros((3, 3))
    base_matrix[:, 0] = base_1
    base_matrix[:, 1] = base_2
    base_matrix[:, 2] = normal_vec

    # need to rotate around z axis, order of basis vectors in hand frame might be switched up
    joint_pos = joint_pos @ base_matrix @ rotation_matrix_z(-np.pi / 2)

    if flip_y_axis:
        joint_pos = joint_pos @ rotation_matrix_y(np.pi)

    if flip_x_axis:
        # flip the x axis
        joint_pos = joint_pos @ rotation_matrix_x(-np.pi / 2)

    rot_matrix = base_matrix

    z_axis = wrist_point - middle_point

    rot_matrix[:, 0] = -base_2
    rot_matrix[:, 1] = -normal_vec
    rot_matrix[:, 2] = base_1
    return joint_pos, rot_matrix
