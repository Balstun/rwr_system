#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
import numpy as np

class RemapperNode(Node):
    def __init__(self):
        super().__init__('remapper_node')
        self._joint_positions_pub = self.create_publisher(
            Float32MultiArray,
            'joint_to_motor_node/joint_positions',
            10
        )
        self._mano_keypoints_sub = self.create_subscription(
            Float32MultiArray,
            '/hand/policy_output',
            self.remap_callback,
            10
        )
        self._wrist_cmd_sub = self.create_subscription(
            Float32,
            '/hand/wrist_cmd',
            self.wrist_cmd_callback,
            10
        )
        self.joint_remapping = {
            0: 13, # pinky_mcp_angle
            1: 14, # pinky_pip_angle
            2: 12, # pinky_abd_angle
            3: 10, # ring_mcp_angle
            4: 11, # ring_pip_angle
            5: 9, # ring_abd_angle
            6: 7, # middle_mcp_angle
            7: 8, # middle_pip_angle
            8: 6, # middle_abd_angle
            9: 4, # index_mcp_angle
            10: 5, # index_pip_angle
            11: 3, # index_abd_angle
            12: 0, # thumb_palm_angle
            13: 1, # thumb_adb_angle
            14: 2, # thumb_mcp_angle (pip?)
        }
        self.num_joints = 16
        self.wrist_cmd = 0.0

    def remap_callback(self, msg):
        target_joint_angles = msg.data
        remapped_joints = self.remap_joints(target_joint_angles)
        self._joint_positions_pub.publish(remapped_joints)

    def remap_joints(self, joints):
        remapped_joints = Float32MultiArray()
        remapped_joints.data = [0.0] * self.num_joints

        for i in range(self.num_joints):
            if i == self.num_joints - 1:
                remapped_joints.data[i] = 0.0 #self.wrist_cmd
            else:
                remapped_joints.data[i] = joints[self.joint_remapping[i]]

        return remapped_joints
    
    def wrist_cmd_callback(self, msg):
        self.wrist_cmd = msg.data

def main(args=None):
    rclpy.init(args=args)
    node = RemapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
