#!/bin/env python3

import numpy as np
import time
from copy import deepcopy

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, TransformStamped, Vector3, Quaternion, PoseStamped
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Float32MultiArray

STATIC_OFFSET = -0.03 # Meters?
JOINT_SUM_THRESHOLD = 7.5 # half of possible total joint angle data
RELEASE_DELAY_THRESHOLD = 100 # 20 Hz * 5 seconds

class ValidateInference(Node):
    def __init__(self):
        super().__init__("validate_inference")

        self.hand_grasping = False
        self.release_delay = 0
        # publishes to franka/end_effector_pose_cmd
        self.arm_publisher = self.create_publisher(
            PoseStamped, "/franka/end_effector_pose_cmd", 10
        )

        self.arm_subscriber = self.create_subscription(
            PoseStamped, "/franka/end_effector_pose_cmd_debug", self.arm_pose_callback, 10
        )
                # Subscriber to 'joint_positions' topic
        self.joint_positions_subscriber = self.create_subscription(
            Float32MultiArray,
            f'{self.get_name()}/joint_positions',
            self.joint_positions_callback,
            10
        )

        self.release_delay = 0

    def arm_pose_callback(self, msg: PoseStamped):
        msg_copy = deepcopy(msg)
        if not self.hand_grasping:
            msg_copy.pose.position.z += STATIC_OFFSET
            if self.release_delay > 0:
                msg_copy.pose.position.z += STATIC_OFFSET / self.release_delay
        self.arm_publisher.publish(msg_copy)

    def joint_positions_callback(self, msg: Float32MultiArray):
        joint_sum = sum(msg.data)

        self.get_logger().info(self.hand_grasping)

        if joint_sum > JOINT_SUM_THRESHOLD:
            self.hand_grasping = True
        else:
            if self.hand_grasping:
                if self.release_delay > RELEASE_DELAY_THRESHOLD:
                    # Ideally the first time it has released the object.
                    self.hand_grasping = False
                    self.release_delay = 0
                else:
                    self.release_delay += 1



def main(args=None):
    rclpy.init(args=args)
    validate_inference = ValidateInference()
    rclpy.spin(validate_inference)
    validate_inference.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
