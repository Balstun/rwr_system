#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
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
        self.joint_remapping = {
            0: 14, # pinky_mcp_angle
            1: 15, # pinky_pip_angle
            2: 13, # pinky_abd_angle
            3: 11, # ring_mcp_angle
            4: 12, # ring_pip_angle
            5: 10, # ring_abd_angle
            6: 8, # middle_mcp_angle
            7: 9, # middle_pip_angle
            8: 7, # middle_abd_angle
            9: 5, # index_mcp_angle
            10: 6, # index_pip_angle
            11: 4, # index_abd_angle
            12: 1, # thumb_palm_angle
            13: 2, # thumb_adb_angle
            14: 3, # thumb_mcp_angle (pip?)
            15: 0, # wrist_angle
        }
        self.num_joints = 16

    def remap_callback(self, msg):
        target_joint_angles = msg.data
        remapped_joints = self.remap_joints(target_joint_angles)
        self._joint_positions_pub.publish(remapped_joints)

    def remap_joints(self, joints):
        remapped_joints = Float32MultiArray()
        remapped_joints.data = [0.0] * self.num_joints

        wrist_enabled = True # Wrist 
        # TODO: Enable as a parameter within the retargeter/scheme

        for i in range(self.num_joints):
            if wrist_enabled:
                remapped_joints.data[i] = joints[self.joint_remapping[i]]
            else:
                if i == self.num_joints - 1:
                    continue
                remapped_joints.data[i] = joints[self.joint_remapping[i] - 1]

        return remapped_joints

def main(args=None):
    rclpy.init(args=args)
    node = RemapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
