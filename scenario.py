import os

tennis_ball_file = os.path.join(os.getcwd(), "object_files/Tennis_ball.sdf")

scenario_data = f"""
directives:
- add_model:
    name: Tennis_ball
    file: file://{tennis_ball_file}
    default_free_body_pose:
        Tennis_ball:
            translation: [1.5, 0, 0.75]
            rotation: !Rpy {{ deg: [42, 33, 18] }}
"""

# --------------------------------- DIRECTIVES ---------------------------------
"""
Notes:
 - If using iiwa7, weld `iiwa_link_0` to world. If using iiwa14, weld `base` to world.
   Only `link` tags can be welded (?)
"""
# scenario_data = f"""
# directives:
# - add_model:
#     name: iiwa
#     file: package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision.urdf
#     default_joint_positions:
#         iiwa_joint_2: [0.5]
#         iiwa_joint_4: [-1.2]
#         iiwa_joint_6: [1.6]
# - add_weld:
#     parent: world
#     child: iiwa::base

# - add_model:
#     name: wsg
#     file: package://drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf
# - add_weld:
#     parent: iiwa::iiwa_link_7
#     child: wsg::body
#     X_PC:
#         translation: [0, 0, 0.09]
#         rotation: !Rpy {{ deg: [90, 0, 90]}}

# - add_model:
#     name: camera0
#     file: package://manipulation/camera_box.sdf
# - add_weld:
#     parent: world
#     child: camera0::base
#     X_PC:
#         translation: [2, 1.5, 1.5]
#         rotation: !Rpy {{deg: [-30, 0, 135]}}
# """

# scenario_data += f"""
# - add_model:
#     name: Tennis_ball
#     file: file://{tennis_ball_file}
#     default_free_body_pose:
#         Tennis_ball:
#             translation: [1.5, 0, 0.75]
#             rotation: !Rpy {{ deg: [42, 33, 18] }}
# """

# scenario_data += """
# model_drivers:
#     iiwa: !IiwaDriver
#       hand_model_name: wsg
#     wsg: !SchunkWsgDriver {}
# """
        
# ---------------------------------- CAMERAS ----------------------------------
# scenario_data += f"""
# cameras:
#     camera0:
#         name: camera0
#         clipping_near: 0.1 # To clip out the camera object
#         X_PB:
#             base_frame: camera0::base
#             rotation: !Rpy {{ deg: [-90, 0, 0]}}
#     camera1:
#         name: camera1
#         clipping_near: 0.1 # To clip out the camera object
#         X_PB:
#             base_frame: camera0::base
#             rotation: !Rpy {{ deg: [-90, 0, 0]}}
# """