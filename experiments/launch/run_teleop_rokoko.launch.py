from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription(
        [

            # Rokoko Ingress
            Node(
                package="ingress",
                executable="rokoko_node.py",
                name="rokoko_node",
                output="log",
                parameters=[
                    {"rokoko_tracker/ip": "0.0.0.0"},
                    {"rokoko_tracker/port": 14043},
                    {"rokoko_tracker/use_coil": True}
                ],
            ),

            # RETARGET NODE
            Node(
                package="retargeter",
                executable="retargeter_node.py",
                name="retargeter_node",
                output="screen",
                parameters=[
                    {
                        "retarget/mjcf_filepath": os.path.join(
                            get_package_share_directory("mujoco_sim"),
                            "no_wrist_mjcf",
                            "Biomimic_hand_job.xml",
                        )
                    },
                    {
                        "retarget/retargeter_cfg": os.path.join(
                            get_package_share_directory("experiments"),
                            "cfgs",
                            "retargeter_cfgs_biomimic.yaml",
                        ),
                    },
                    {
                        "retarget/mano_adjustments": os.path.join(
                            get_package_share_directory("experiments"),
                            "cfgs",
                            "retargeter_adjustment.yaml"
                        ),
                    },
                    {"retarget/hand_scheme": "biomimic"},
                    {"debug": True},
                ],
            ),

            # WRIST CONTROLLER NODE
            Node(
                package="wrist_retargeter",
                executable="wrist_retargeter",
                name="wrist_retargeter",
                output="screen",
            ),

            # REMAPPER NODE
            Node(
                package="retargeter",
                executable="remapper_node.py",
                name="remapper",
                output="screen"
            ),
        ]
    )
