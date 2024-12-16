#!/bin/env python3

from time import sleep
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import TransformStamped, PoseStamped
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from threading import Lock
from custom_interfaces.srv import GetSegmentationMask
from typing import Tuple

import torch
import yaml
from faive_system.src.common.utils import numpy_to_float32_multiarray, float32_multiarray_to_numpy
from srl_il.export.il_policy import get_policy_from_ckpt

class CameraListener(Node):
    def __init__(self, camera_topic, name, node):
        self.camera_topic = camera_topic
        self.lock = Lock()
        self.image = None
        self.name = name
        self.im_subscriber = node.create_subscription(
            Image, self.camera_topic, self.recv_im, 10
        )

    def recv_im(self, msg: Image):
        with self.lock:
            self.image = msg

    def get_im(self):
        with self.lock:
            return deepcopy(self.image)

class PolicyPlayerAgent(Node):
    def __init__(self):
        super().__init__("policy_publisher")
        
        self.declare_parameter("camera_topics", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("camera_names", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("policy_ckpt_path", "")   # assume the policy ckpt is saved with its config
        self.declare_parameter("no_segmentation_policy_ckpt_path", "")   # assume the policy ckpt is saved with its config
        self.declare_parameter("hand_qpos_dim", 16) # The dimension of the hand_qpos, we need this because we need to broadcast an all zero command to the hand at the beginning
        self.camera_topics = self.get_parameter("camera_topics").value
        self.camera_names = self.get_parameter("camera_names").value
        self.policy_ckpt_path = self.get_parameter("policy_ckpt_path").value
        self.hand_qpos_dim = self.get_parameter("hand_qpos_dim").value

        self.thumb_data = None
        self.index_data = None
        self.middle_data = None
        self.ring_data = None
        self.pinky_data = None
        self.use_segmentation_policy = True
        self.tray_segmentation_image = None
        
        self.lock = Lock()

        self.hand_pub = self.create_publisher(
            Float32MultiArray, "/joint_to_motor_node/joint_positions", 10
        )
        self.hand_sub = self.create_subscription(
            Float32MultiArray, "/joint_to_motor_node/joint_positions", self.hand_callback, 10
        )
        
        self.arm_publisher = self.create_publisher(
            PoseStamped, "/franka/end_effector_pose_cmd", 10
        )
        self.arm_subscriber = self.create_subscription(
            PoseStamped, "/franka/end_effector_pose", self.arm_pose_callback, 10
        )

        self.sensor_thumb = self.create_subscription(
            Float32, "/thumb_sensor_filtered", self.thumb_cb, 10
        )

        self.sensor_index = self.create_subscription(
            Float32, "/index_sensor_filtered", self.index_cb, 10
        )

        self.sensor_middle = self.create_subscription(
            Float32, "/middle_sensor_filtered", self.middle_cb, 10
        )

        self.sensor_ring = self.create_subscription(
            Float32, "/ring_sensor_filtered", self.ring_cb, 10
        )

        self.sensor_pinky = self.create_subscription(
            Float32, "/pinky_sensor_filtered", self.pinky_cb, 10
        )

        self.camera_listeners = [
            CameraListener(camera_topic, camera_name, self) 
            for camera_topic, camera_name in zip(self.camera_topics, self.camera_names)
        ]
        
        # self.segmentation_client = self.create_client(AddTwoInts, 'add_two_ints')
        # while not self.cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('service not available, waiting again...')
        # self.req = AddTwoInts.Request()

        # def send_request(self, a, b):
        #     self.req.a = a
        #     self.req.b = b
        #     self.future = self.cli.call_async(self.req)
        #     rclpy.spin_until_future_complete(self, self.future)
        #     return self.future.result()

        self.tray_segmentation_mask_publisher = self.create_publisher(
            Image, "/segmentation/tray_segmentation_mask", 10
        )

        self.publish_segmentation_mask_timer = self.create_timer(0.1, self.segmentation_mask_timer_callback)

        self.bridge = CvBridge()
        self.current_wrist_state = None
        self.current_hand_state = None

        if self.use_segmentation_policy:
            self.policy = get_policy_from_ckpt(self.policy_ckpt_path)
        else:
            self.policy = get_policy_from_ckpt(self.no_segmentation_policy_ckpt_path)
        self.policy.reset_policy()
        self.policy_run = self.create_timer(0.05, self.run_policy_cb) # 20hz


        hand_msg = numpy_to_float32_multiarray(np.zeros(self.hand_qpos_dim))
        self.hand_pub.publish(hand_msg)

        input("Press enter to move the robot to the initial pose")
        
        franka_init_pose = PoseStamped()
        franka_init_pose.pose.position = [0.33303902877539454, -0.5352327819313111, 0.460122887480905]
        franka_init_pose.pose.orientation = [0.9123890032495732, 0.0068343934713917575, 0.0034454251504201405, -0.409252644292815]
        franka_init_pose.header.stamp = self.get_clock().now().to_msg()
        franka_init_pose.header.frame_id = "panda_link0"
        self.arm_publisher.publish(franka_init_pose)


    def get_segmentation_masks(self) -> Tuple[Image, Image, Image]:
        client = self.create_client(GetSegmentationMask, "/segment_svc")
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        request = GetSegmentationMask.Request()

        resp: GetSegmentationMask.Response = client.call(request)

        if resp.success:
            self.tray_segmentation_image = resp.tray_mask.data
        else:
            self.use_segmentation_policy = False
        raise Exception("Error in calling segmentation service")
    
    def segmentation_mask_timer_callback(self):
        if self.use_segmentation_policy:
            self.tray_segmentation_mask_publisher.publish(self.tray_segmentation_image)

    
    def publish(self, wrist_policy: np.ndarray, hand_policy: np.ndarray):
        # publish hand policy
        hand_msg = numpy_to_float32_multiarray(hand_policy)
        self.hand_pub.publish(hand_msg)

        # publish wrist policy
        wrist_msg = PoseStamped()
        wrist_msg.pose.position.x, wrist_msg.pose.position.y, wrist_msg.pose.position.z = wrist_policy[:3].astype(np.float64)
        (   wrist_msg.pose.orientation.x,
            wrist_msg.pose.orientation.y,
            wrist_msg.pose.orientation.z,
            wrist_msg.pose.orientation.w,
        ) = wrist_policy[3:].astype(np.float64)
        wrist_msg.header.stamp = self.get_clock().now().to_msg()   
        wrist_msg.header.frame_id = "panda_link0" 
        self.arm_publisher.publish(wrist_msg)
    
    def arm_pose_callback(self, msg: PoseStamped):
        current_wrist_state_msg = msg.pose
        position = [current_wrist_state_msg.position.x, current_wrist_state_msg.position.y, current_wrist_state_msg.position.z]
        quaternion = [current_wrist_state_msg.orientation.x, current_wrist_state_msg.orientation.y, current_wrist_state_msg.orientation.z, current_wrist_state_msg.orientation.w]
        self.current_wrist_state = np.concatenate([position, quaternion])
    
    def hand_callback(self, msg: Float32MultiArray):
        self.current_hand_state = float32_multiarray_to_numpy(msg)

    def get_current_observations(self):
        obs_dict = {}
        get_data_success = True
        
        images = {camera.name: camera.get_im() for camera in self.camera_listeners}
        if any([im is None for im in images.values()]):
            get_data_success = False
            print("Missing camera images", [im is not None for im in images.values()])
            return get_data_success, obs_dict

        images = {
            k: self.bridge.imgmsg_to_cv2(v, "bgr8").transpose(2, 0, 1)/255.0
            for k, v in images.items()
        }

        with self.lock:
            qpos_franka = self.current_wrist_state
            qpos_hand = self.current_hand_state
        if qpos_franka is None or qpos_hand is None:
            print("missing qpos_franka", qpos_franka is None)
            print("missing qpos_hand", qpos_hand is None)
            return False, obs_dict

        thumb_data = self.thumb_data
        index_data = self.index_data
        middle_data = self.middle_data
        ring_data = self.ring_data
        pinky_data = self.pinky_data

        sensor_obs = {"thumb_sensor": thumb_data, "index_sensor": index_data, "middle_sensor": middle_data, "ring_sensor": ring_data, "pinky_sensor": pinky_data}

        for o in sensor_obs.values():
            if o is None:
                print("Missing sensor data")
                return False, obs_dict

        obs_dict.update(images)
        obs_dict['qpos_franka'] = qpos_franka
        obs_dict['qpos_hand'] = qpos_hand
        obs_dict.update(sensor_obs)
        if self.use_segmentation_policy:
            obs_dict['tray_segmentation_image'] = self.tray_segmentation_image
        return get_data_success, obs_dict
    
    def run_policy_cb(self):

        get_data_success, obs_dict = self.get_current_observations()
        if not get_data_success:
            self.get_logger().info("No observations available. Sleeping for 1 seconds.")
            sleep(1)
            return
        with torch.inference_mode():
            obs_dict = {k: torch.tensor(v).float().unsqueeze(0) for k, v in obs_dict.items()} # add batch dimension
            actions = self.policy.predict_action(obs_dict)
            wrist_action = actions["actions_franka"][0].cpu().numpy()
            hand_action = actions["actions_hand"][0].cpu().numpy()
        self.publish(wrist_action, hand_action)

    def thumb_cb(self, msg: Float32):
        self.thumb_data = np.array([msg.data])

    def index_cb(self, msg: Float32):
        self.index_data = np.array([msg.data])

    def middle_cb(self, msg: Float32):
        self.middle_data = np.array([msg.data])

    def ring_cb(self, msg: Float32):
        self.ring_data = np.array([msg.data])

    def pinky_cb(self, msg: Float32):
        self.pinky_data = np.array([msg.data])


def main(args=None):
    rclpy.init(args=args)
    node = PolicyPlayerAgent()
    node.get_segmentation_masks()
    rclpy.spin(node)
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()
