#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
import numpy as np
from typing import Callable
from functools import wraps
from faive_system.src.common.subsystem_poller import SubsystemPoller

def check_subsystem_enabled(func: Callable):
    """
    Decorator to check if node is enabled
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.enabled:
            return func(self, *args, **kwargs)
        else:
            self.get_logger().warn(f"{func.__name__} not performed because node is not enabled.", skip_first=False, throttle_duration_sec=1.0)
            return None
    return wrapper

class RemapperNode(Node):
    def __init__(self):
        super().__init__('remapper_node')
        
        self.subsystem_poller = SubsystemPoller(self, "retargeter_enabled")
        self.enabled = False

        self._joint_positions_pub = self.create_publisher(
            Float32MultiArray,
            'joint_to_motor_node/joint_positions',
            10
        )

        self.num_joints = 16

        self.remapped_joint_angles = np.zeros(self.num_joints - 1)
        self.wrist_cmd = 0.0
        
        self.timer_ = self.create_timer(0.01, self.pub_joint_angles)

        self._mano_keypoints_sub = self.create_subscription(
            Float32MultiArray,
            '/hand/policy_output',
            self.remap_joints,
            10
        )
        self._wrist_cmd_sub = self.create_subscription(
            Float32,
            '/hand/wrist_pitch_cmd',
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


    @check_subsystem_enabled
    def pub_joint_angles(self):
        msg = Float32MultiArray()
        data = np.hstack((self.remapped_joint_angles, np.array(self.wrist_cmd)))
        msg.data = data.tolist()

        self._joint_positions_pub.publish(msg)

    def remap_joints(self, msg: Float32MultiArray):
        joints = msg.data

        remapped_joint_angles = [0.0] * (self.num_joints - 1)
        for i in range(len(joints)):
            remapped_joint_angles[i] = joints[self.joint_remapping[i]]

        self.remapped_joint_angles = np.array(remapped_joint_angles)
    
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
