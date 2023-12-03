"""
Note:

For debugging optimization problems/mathematical programs in drake, you can view which constraints are being violated with "GetInfeasibleConstraintNames" and "GetInfeasibleConstraints"

https://drake.mit.edu/doxygen_cxx/classdrake_1_1solvers_1_1_mathematical_program_result.html
"""


import time

import numpy as np
from pydrake.all import (
    MultibodyPlant,
    BsplineTrajectory,
    KinematicTrajectoryOptimization,
    MinimumDistanceLowerBoundConstraint,
    Parser,
    PositionConstraint,
    SpatialVelocityConstraint,
    RigidTransform,
    Solve,
    Sphere,
    Rgba,
    Quaternion
)

from manipulation.meshcat_utils import PublishPositionTrajectory, AddMeshcatTriad
from manipulation.scenarios import AddIiwa, AddPlanarIiwa, AddShape, AddWsg
from manipulation.utils import ConfigureParser

def motion_test(meshcat, obj_traj, obj_catch_t):

    # Setup a new MBP with just the iiwa which the KinematicTrajectoryOptimization will use
    plant = MultibodyPlant(0.0)
    iiwa_file = "package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision.urdf"
    iiwa = Parser(plant).AddModelsFromUrl(iiwa_file)[0]  # ModelInstance object
    # wsg_file = "package://drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf"
    # wsg = Parser(plant).AddModelsFromUrl(wsg_file)[0]  # ModelInstance object
    world_frame = plant.world_frame()
    base_frame = plant.GetFrameByName("base")
    gripper_frame = plant.GetFrameByName("iiwa_link_7")
    # gripper_frame = plant.GetFrameByName("body")
    plant.WeldFrames(world_frame, base_frame)  # Weld iiwa to world
    # plant.WeldFrames(l7_frame, gripper_frame)  # Weld wsg to iiwa
    plant.Finalize()
    plant_context = plant.CreateDefaultContext()

    # Plot spheres to visualize obj trajectory
    times = np.linspace(0, 1, 50)
    for t in times:
        obj_pose_data = obj_traj.value(t)  # (7,1) vector containing x,y,z,q0,q1,q2,q3
        obj_pose_quaterion = obj_pose_data[3:]
        obj_pose_position = obj_pose_data[:3]
        obj_pose = RigidTransform(Quaternion(obj_pose_quaterion), obj_pose_position)
        meshcat.SetObject(str(t), Sphere(0.005), Rgba(0.4, 1, 1, 1))
        meshcat.SetTransform(str(t), obj_pose)

    X_WStart = plant.CalcRelativeTransform(plant_context, world_frame, gripper_frame)  # robot current pose
    obj_catch_point = obj_traj.value(obj_catch_t)  # (7,) np array
    obj_catch_quaterion = obj_catch_point[3:]
    obj_catch_position = obj_catch_point[:3]
    X_WGoal = RigidTransform(Quaternion(obj_catch_quaterion), obj_catch_position)
    print(f"X_WStart: {X_WStart}")
    print(f"X_WGoal: {X_WGoal}")

    AddMeshcatTriad(meshcat, "start", X_PT=X_WStart, opacity=0.5)
    meshcat.SetTransform("start", X_WStart)
    AddMeshcatTriad(meshcat, "goal", X_PT=X_WGoal, opacity=0.5)
    meshcat.SetTransform("goal", X_WGoal)

    num_q = plant.num_positions()  # =7 (all of iiwa's joints)
    q0 = plant.GetPositions(plant_context)

    trajopt = KinematicTrajectoryOptimization(num_q, 10)  # 10 control points in Bspline
    prog = trajopt.get_mutable_prog()

    # Guess 10 control points in 7D
    q_guess = np.tile(q0.reshape((7, 1)), (1, trajopt.num_control_points()))  # (7,10) np array
    q_guess[0, :] = np.linspace(0, -np.pi / 2, trajopt.num_control_points())
    path_guess = BsplineTrajectory(trajopt.basis(), q_guess)
    trajopt.SetInitialGuess(path_guess)

    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(1.0)
    trajopt.AddPositionBounds(
        plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    )
    trajopt.AddVelocityBounds(
        plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits()
    )

    trajopt.AddDurationConstraint(0.5, 50)

    # start constraint
    start_constraint = PositionConstraint(
        plant,
        world_frame,
        X_WStart.translation(),  # upper limit
        X_WStart.translation(),  # lower limit
        gripper_frame,
        [0, 0.1, 0],
        plant_context,
    )
    trajopt.AddPathPositionConstraint(start_constraint, 0)
    prog.AddQuadraticErrorCost(
        np.eye(num_q), q0, trajopt.control_points()[:, 0]
    )

    # goal constraint
    goal_constraint = PositionConstraint(
        plant,
        world_frame,
        X_WGoal.translation(),  # upper limit
        X_WGoal.translation(),  # lower limit
        gripper_frame,
        [0, 0.1, 0],
        plant_context,
    )
    trajopt.AddPathPositionConstraint(goal_constraint, 1)
    prog.AddQuadraticErrorCost(
        np.eye(num_q), q0, trajopt.control_points()[:, -1]
    )

    # start with zero velocity
    trajopt.AddPathVelocityConstraint(
        np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0
    )
    # end with velocity equal to object's velocity at that moment
    obj_vel_at_catch = obj_traj.EvalDerivative(obj_catch_t)[:3]  # (3,) np array

    final_vel_constraint = SpatialVelocityConstraint(
        plant,
        world_frame,
        obj_vel_at_catch,  # upper limit
        obj_vel_at_catch,  # lower limit
        gripper_frame,
        [0, 0, 0],
        plant_context,
    )
    trajopt.AddVelocityConstraintAtNormalizedTime(final_vel_constraint, 1)

    # collision constraints
    # collision_constraint = MinimumDistanceLowerBoundConstraint(
    #     plant, 0.001, plant_context, None, 0.01
    # )
    # evaluate_at_s = np.linspace(0, 1, 50)
    # for s in evaluate_at_s:
    #     trajopt.AddPathPositionConstraint(collision_constraint, s)

    result = Solve(prog)
    if not result.is_success():
        print("Trajectory optimization failed")
        print(result.get_solver_id().name())

    print(f"result.GetSolution(): {result.GetSolution()}")
    final_traj = trajopt.ReconstructTrajectory(result)  # BSplineTrajectory
    print(f"final_traj.value(0): {final_traj.value(0)}")