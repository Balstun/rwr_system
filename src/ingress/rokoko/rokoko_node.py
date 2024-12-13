#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from rokoko_ingress import RokokoTracker
from faive_system.src.common.utils import numpy_to_float32_multiarray    
from faive_system.src.common.subsystem_poller import SubsystemPoller
from functools import wraps
from typing import Callable


class RokokoNode(Node):
    def __init__(self, debug=False):
        super().__init__("rokoko_node")

        self.subsystem_poller = SubsystemPoller(self, "rokoko_enabled")
        self.enabled = False

        # start tracker
        self.declare_parameter("rokoko_tracker/ip", "0.0.0.0")
        self.declare_parameter("rokoko_tracker/port", 14043)
        self.declare_parameter("rokoko_tracker/use_coil", True)

        ip = self.get_parameter("rokoko_tracker/ip").value
        port = self.get_parameter("rokoko_tracker/port").value
        self.use_coil = self.get_parameter("rokoko_tracker/use_coil").value

        self.tracker = RokokoTracker(ip=ip, port=port, use_coil=self.use_coil)
        self._logger.info("RokokoTracker launched")

        self.tracker.start()

        ingress_period = 0.005  # Timer period in seconds
        self.timer = self.create_timer(ingress_period, self.timer_publish_cb)

        self.ingress_mano_pub = self.create_publisher(
            Float32MultiArray, "/ingress/mano", 10
        )
        self.ingress_wrist_pub = self.create_publisher(
            PoseStamped, "/ingress/wrist", 10
        )

        self.ingress_right_lower_arm_pub = self.create_publisher(
            PoseStamped, "/ingress/right_lower_arm", 10
        )
        self.ingress_elbow_pub = self.create_publisher(
            PoseStamped, "/ingress/elbow", 10
        )

        self.debug = debug


    def check_subsystem_enabled(self, func: Callable):
        """
        Decorator to check if node is enabled
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.enabled:
                return func(*args, **kwargs)
            else:
                self.get_logger().warn(f"{func.__name__} not performed because rokoko ingress is not enabled.", throttle_duration_sec=2.0)
                return None
        return wrapper

    def timer_publish_cb(self):

        # self._logger.info("Current debug flag: {}".format(self.tracker.debug_flag))

        key_points = self.tracker.get_keypoint_positions()
        wait_cnt = 1
        while (key_points is None):
            if (not (wait_cnt % 100000)):
                print("waiting for hand tracker", wait_cnt//100000)
            wait_cnt+=1
            key_points = self.tracker.get_keypoint_positions()
        keypoint_positions, timestamp = key_points

        keypoint_positions_msg = numpy_to_float32_multiarray(keypoint_positions)

        self.check_subsystem_enabled(self.ingress_mano_pub.publish)(keypoint_positions_msg)

        if self.use_coil:
            wrist_pos, wrist_rot = self.tracker.get_wrist_pose()
            # Create a PoseStamped message
            right_lower_arm_msg = PoseStamped()
            right_lower_arm_msg.header.frame_id = "coil"
            right_lower_arm_msg.header.stamp = self.get_clock().now().to_msg()

            # Assign position using Point
            right_lower_arm_msg.pose.position = Point(
                x=wrist_pos[0], y=wrist_pos[1], z=wrist_pos[2]
            )

            # Assign orientation using Quaternion
            right_lower_arm_msg.pose.orientation = Quaternion(
                x=wrist_rot[0], y=wrist_rot[1], z=wrist_rot[2], w=wrist_rot[3]
            )

            # Publish the message
            self.check_subsystem_enabled(self.ingress_wrist_pub.publish)(right_lower_arm_msg)

            # Do the same for right_lower_arm ------------------------------------------------

            right_lower_arm_pos, right_lower_arm_quat = self.tracker.get_right_lower_arm_pose()
            
            # Create a PoseStamped message
            right_lower_arm_msg = PoseStamped()
            right_lower_arm_msg.header.frame_id = "coil"
            right_lower_arm_msg.header.stamp = self.get_clock().now().to_msg()

            # Assign position using Point
            right_lower_arm_msg.pose.position = Point(
                x=right_lower_arm_pos[0], y=right_lower_arm_pos[1], z=right_lower_arm_pos[2]
            )

            # Assign orientation using Quaternion
            right_lower_arm_msg.pose.orientation = Quaternion(
                x=right_lower_arm_quat[0], y=right_lower_arm_quat[1], z=right_lower_arm_quat[2], w=right_lower_arm_quat[3]
            )

            # Publish the message
            self.check_subsystem_enabled(self.ingress_right_lower_arm_pub.publish)(right_lower_arm_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RokokoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
