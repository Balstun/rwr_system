#!/usr/bin/env python3
from numpy.typing import NDArray
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32, String
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from faive_system.src.retargeter import Retargeter
from faive_system.src.common.utils import numpy_to_float32_multiarray
import os
from faive_system.src.viz.visualize_mano import ManoHandVisualizer
from std_msgs.msg import ColorRGBA

class RetargeterNode(Node):
    def __init__(self, debug=False):
        super().__init__("rokoko_node")

        # start retargeter
        self.declare_parameter("retarget/mjcf_filepath", rclpy.Parameter.Type.STRING)
        self.declare_parameter("retarget/urdf_filepath", rclpy.Parameter.Type.STRING)
        self.declare_parameter("retarget/hand_scheme", rclpy.Parameter.Type.STRING)
        self.declare_parameter("debug", rclpy.Parameter.Type.BOOL)

        try:
            mjcf_filepath = self.get_parameter("retarget/mjcf_filepath").value
        except:
            mjcf_filepath = None

        try:
            urdf_filepath = self.get_parameter("retarget/urdf_filepath").value
        except:
            urdf_filepath = None
        hand_scheme = self.get_parameter("retarget/hand_scheme").value
        debug = self.get_parameter("debug").value

        # subscribe to ingress topics
        self.ingress_mano_sub = self.create_subscription(
            Float32MultiArray, "/ingress/mano", self.ingress_mano_cb, 10
        )

        self.retargeter = Retargeter(
            device="cpu",  mjcf_filepath= mjcf_filepath, urdf_filepath=urdf_filepath, hand_scheme=hand_scheme
        )

        self.joints_pub = self.create_publisher(
            Float32MultiArray, "/hand/policy_output", 10
        )
        self.mano_vec_pub = self.create_publisher(
            Marker, "/mano/keyvecs", 10
        )
        self.mujoco_vec_pub = self.create_publisher(
            Float32MultiArray, "/retarget/mujoco_keyvecs", 10
        )

        self.debug = debug
        if self.debug:
            self.rviz_pub = self.create_publisher(MarkerArray, 'retarget/normalized_mano_points', 10)
            self.mano_hand_visualizer = ManoHandVisualizer(self.rviz_pub, hand_scheme)


        self.timer = self.create_timer(0.005, self.timer_publish_cb)

    def ingress_mano_cb(self, msg):
        self.keypoint_positions = np.array(msg.data).reshape(-1, 3)


    def timer_publish_cb(self):
        try:
            if self.debug:
                self.mano_hand_visualizer.reset_markers()

            debug_dict = {}
            joint_angles, debug_dict = self.retargeter.retarget(self.keypoint_positions, debug_dict)


            if "keyvec_mano" in debug_dict.keys():
                default_color = (0, 1, 0, 1)
                # Prepare the list of vectors
                keyvec_mano = debug_dict["keyvec_mano"]
                start, end = keyvec_mano["start"], keyvec_mano["end"].values()

                if len(start) == 1:
                    vectors = [(start[0].numpy(), e.numpy(), default_color) for e in end]
                else:
                    vectors = [(s.numpy(), e.numpy(), default_color) for s, e in zip(start, end)]

                self.publish_marker(vectors)

            if "keyvec_mujoco" in debug_dict.keys():
                keyvec_mujoco = debug_dict["keyvec_mujoco"]
                start, end = keyvec_mujoco["start"][0].detach().cpu(), [t.detach().cpu().numpy() for t in keyvec_mujoco["end"].values()]
                
                st_np_arr = np.tile(start, (5, 1)).reshape((-1, 3))
                end_np_arr = np.array(end).reshape((-1, 3))
                mujoco_vecs = np.hstack((st_np_arr, end_np_arr))
                self.publish_mujoco_vecs(mujoco_vecs)


            if self.debug:
                self.mano_hand_visualizer.generate_hand_markers(
                    debug_dict["normalized_joint_pos"],
                    stamp=self.get_clock().now().to_msg(),
                )

            self.joints_pub.publish(
                numpy_to_float32_multiarray(np.deg2rad(joint_angles))
            )

            if self.debug:
                self.mano_hand_visualizer.publish_markers()
        except Exception as e:
            print(f"Error in timer_publish_cb: {e}")
            pass

    def publish_mujoco_vecs(self, mujoco_vecs: NDArray[np.float32]):
        self.mujoco_vec_pub.publish(numpy_to_float32_multiarray(mujoco_vecs))

    def publish_marker(self, vectors):
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'vectors'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01  # Line width

        # Default color
        default_color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)  # Blue

        for vector in vectors:
            start, end, color_override = vector
            start = start.squeeze().tolist()
            end = end.squeeze().tolist()
            start_point = Point(x=start[0], y=start[1], z=start[2])
            end_point = Point(x=end[0], y=end[1], z=end[2])
            marker.points.append(start_point)
            marker.points.append(end_point)

            # Handle color override or use default
            marker.colors.append(default_color)
            marker.colors.append(default_color)

        self.mano_vec_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = RetargeterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
