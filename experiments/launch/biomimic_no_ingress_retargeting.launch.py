from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

"""
Use this when you're playing back /ingress/mano ROS Bags
"""
def generate_launch_description():
    urdf = os.path.join(
    get_package_share_directory('viz'),
    "models",
    "biomimic_hand_v3",
    "urdf",
    "biomimic_hand.urdf")

    with open(urdf, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription(
        [
            # RETARGET NODE
            Node(
                package="retargeter",
                executable="retargeter_node.py",
                name="retargeter_node",
                output="screen",
                parameters=[
                    {
                        "retarget/mjcf_filepath": os.path.join(
                            get_package_share_directory("viz"),
                            "models",
                            "biomimic_hand_v3",
                            "Biomimic_hand_job.xml",
                        )
                    },
                    {"retarget/hand_scheme": "biomimic"},
                    {"debug": True},
                ],
            ),

            # REMAPPER NODE
            Node(
                package="retargeter",
                executable="remapper_node.py",
                name="remapper",
                output="screen"
            ),

#             Node(
#                 package='robot_state_publisher',
#                 executable='robot_state_publisher',
#                 name='robot_state_publisher',
#                 output='screen',
#                 parameters=[{'robot_description': robot_desc,}],
#                 arguments=[urdf]),

            # Node(
            #     package='rviz2',
            #     executable='rviz2',
            #     name='rviz2',
            #     output='screen',
            #     arguments=['-d', os.path.join(get_package_share_directory('viz'), 'rviz', 'retarget_config.rviz')],
            #     ),

        ]
    )
