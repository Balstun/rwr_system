import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

class RemapperNode(Node):
    def __init__(self):
        super().__init__('remapper node')
        self._joint_positions_pub = self.create_publisher(
            Float32MultiArray,
            'joint_positions',
            10
        )
        self._mano_keypoints_sub = self.create_subscription(
            Float32MultiArray,
            '/hand/policy_output',
            self.remap_callback,
            10
        )
        self.joint_remapping = {
            0: 18, #pinky_mcp_angle
            1: 19, #pinky_pip_angle
            2: 17, #pinky_abd_angle #TODO: Adjust to be keypoint vector
            3: 14, #ring_mcp_angle
            4: 15, #ring_pip_angle
            5: 13, #ring_abd_angle #TODO: Adjust to be keypoint vector
            6: 11, #middle_mcp_angle
            7: 10, #middle_pip_angle
            8: 12, #middle_abd_angle #TODO: Adjust to be keypoint vector
            9: 6, #index_mcp_angle
            10: 7, #index_pip_angle
            11: 5, #index_abd_angle #TODO: Adjust to be keypoint vector
            12: 2, #thumb_mcp_angle
            13: 3, #thumb_pip_angle
            14: 1, #thumb_abd_angle #TODO: Adjust to be keypoint vector
        }

    def remap_callback(self, msg):
        mano_keypoints = msg.data
        remapped_joints = self.remap_joints(mano_keypoints)
        self.pub.publish(remapped_joints)

    def remap_joints(self, joints):
        remapped_joints = Float32MultiArray()
        remapped_joints = [0] * 16

        for i in range(len(remapped_angles)):
            remapped_joints = joints[self.joint_remapping[i]]

        return remapped_joints

