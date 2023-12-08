"""
Note:

For debugging optimization problems/mathematical programs in drake, you can view which constraints are being violated with "GetInfeasibleConstraintNames" and "GetInfeasibleConstraints"

https://drake.mit.edu/doxygen_cxx/classdrake_1_1solvers_1_1_mathematical_program_result.html
"""


import time

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    BsplineTrajectory,
    KinematicTrajectoryOptimization,
    MinimumDistanceLowerBoundConstraint,
    Parser,
    PositionConstraint,
    OrientationConstraint,
    SpatialVelocityConstraint,
    RigidTransform,
    Solve,
    Sphere,
    Rgba,
    Quaternion,
    RotationMatrix,
)

from manipulation.meshcat_utils import AddMeshcatTriad

def add_constraints(plant, 
                    plant_context, 
                    plant_auto_diff, 
                    trajopt, 
                    prog, 
                    world_frame, 
                    gripper_frame, 
                    X_WStart, 
                    X_WGoal, 
                    num_q, 
                    q0, 
                    obj_traj, 
                    obj_catch_t,
                    duration_cost=50.0,
                    duration_constraint = -1,
                    acceptable_pos_err=0.0,
                    theta_bound = 0.05,
                    acceptable_vel_err=0.05):
    """
    Relevant Constraints who have tunable acceptable error measurements.
    """
    trajopt.AddDurationCost(duration_cost)  # increase to make iiwa faster

    if (duration_constraint == -1):
        trajopt.AddDurationConstraint(0.5, 50)
    else:
        trajopt.AddDurationConstraint(duration_constraint, duration_constraint)

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
    goal_pos_constraint = PositionConstraint(
        plant,
        world_frame,
        X_WGoal.translation() - acceptable_pos_err,  # upper limit
        X_WGoal.translation() + acceptable_pos_err,  # lower limit
        gripper_frame,
        [0, 0, 0.1],
        plant_context,
    )
    goal_orientation_constraint = OrientationConstraint(
        plant,
        world_frame,
        X_WGoal.rotation(),  # orientation of gripper in world frame ...
        gripper_frame,
        RotationMatrix(),  # ... must equal origin in gripper frame
        theta_bound,
        plant_context
    )
    trajopt.AddPathPositionConstraint(goal_pos_constraint, 1)
    trajopt.AddPathPositionConstraint(goal_orientation_constraint, 1)
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
        plant_auto_diff,
        plant_auto_diff.world_frame(),
        obj_vel_at_catch - acceptable_vel_err,  # upper limit
        obj_vel_at_catch + acceptable_vel_err,  # lower limit
        plant_auto_diff.GetFrameByName("iiwa_link_7"),
        np.array([0, 0, 0]).reshape(-1,1),
        plant_auto_diff.CreateDefaultContext(),
    )

    # collision constraints
    # collision_constraint = MinimumDistanceLowerBoundConstraint(
    #     plant, 0.001, plant_context, None, 0.01
    # )
    # evaluate_at_s = np.linspace(0, 1, 50)
    # for s in evaluate_at_s:
    #     trajopt.AddPathPositionConstraint(collision_constraint, s)

    return final_vel_constraint


def motion_test(original_plant, meshcat, obj_traj, obj_catch_t):
    original_plant_positions = original_plant.GetPositions(original_plant.CreateDefaultContext(), original_plant.GetModelInstanceByName("iiwa"))

    # Setup a new MBP with just the iiwa which the KinematicTrajectoryOptimization will use
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
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

    # Set positions of this new plant equal to the positions of the main plant
    plant.SetPositions(plant_context, iiwa, original_plant_positions)

    # Create auto-differentiable version the plant in order to set velocity constraints
    plant_auto_diff = plant.ToAutoDiffXd()

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
    # print(f"X_WStart: {X_WStart}")
    # print(f"X_WGoal: {X_WGoal}")

    AddMeshcatTriad(meshcat, "motion_test_start", X_PT=X_WStart, opacity=0.5)
    meshcat.SetTransform("motion_test_start", X_WStart)
    AddMeshcatTriad(meshcat, "motion_test_goal", X_PT=X_WGoal, opacity=0.5)
    meshcat.SetTransform("motion_test_goal", X_WGoal)

    num_q = plant.num_positions()  # =7 (all of iiwa's joints)
    q0 = plant.GetPositions(plant_context)

    trajopt = KinematicTrajectoryOptimization(num_q, 10)  # 10 control points in Bspline
    prog = trajopt.get_mutable_prog()

    # Guess 10 control points in 7D
    q_guess = np.tile(q0.reshape((7, 1)), (1, trajopt.num_control_points()))  # (7,10) np array
    q_guess[0, :] = np.linspace(0, -np.pi / 2, trajopt.num_control_points())
    path_guess = BsplineTrajectory(trajopt.basis(), q_guess)
    trajopt.SetInitialGuess(path_guess)

    trajopt.AddPathLengthCost(1.0)
    trajopt.AddPositionBounds(
        plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    )
    trajopt.AddVelocityBounds(
        plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits()
    )

    final_vel_constraint = add_constraints(plant, 
                                           plant_context, 
                                           plant_auto_diff, 
                                           trajopt, 
                                           prog, 
                                           world_frame, 
                                           gripper_frame, 
                                           X_WStart, 
                                           X_WGoal, 
                                           num_q, 
                                           q0, 
                                           obj_traj, 
                                           obj_catch_t,
                                           duration_cost=1.0,
                                           duration_constraint=-1,
                                           acceptable_pos_err=0.4,
                                           theta_bound = 0.5,
                                           acceptable_vel_err=3.0)
    
    # For whatever reason, running AddVelocityConstraintAtNormalizedTime inside the function above causes segfault with no error message.
    trajopt.AddVelocityConstraintAtNormalizedTime(final_vel_constraint, 1)

    # First solve with looser constraints
    result = Solve(prog)
    if not result.is_success():
        print("ERROR: First Trajectory optimization failed: " + str(result.get_solver_id().name()))
    else:
        print("First solve succeeded.")
    solved_traj = trajopt.ReconstructTrajectory(result)  # BSplineTrajectory
    
    # Try again but with tighter constraints and using the last attempt as an initial guess
    trajopt_refined = KinematicTrajectoryOptimization(num_q, 10)  # 10 control points in Bspline
    prog_refined = trajopt_refined.get_mutable_prog()
    final_vel_constraint = add_constraints(plant, 
                                           plant_context, 
                                           plant_auto_diff, 
                                           trajopt_refined, 
                                           prog_refined, 
                                           world_frame, 
                                           gripper_frame, 
                                           X_WStart, 
                                           X_WGoal, 
                                           num_q, 
                                           q0, 
                                           obj_traj, 
                                           obj_catch_t,
                                           duration_constraint=1)
    # For whatever reason, running AddVelocityConstraintAtNormalizedTime inside the function above causes segfault with no error message.
    trajopt_refined.AddVelocityConstraintAtNormalizedTime(final_vel_constraint, 1)

    trajopt_refined.SetInitialGuess(solved_traj)
    result = Solve(prog_refined)
    if not result.is_success():
        print("ERROR: Second Trajectory optimization failed: " + str(result.get_solver_id().name()))
    else:
        print("Second solve succeeded.")

    final_traj = trajopt_refined.ReconstructTrajectory(result)  # BSplineTrajectory

    return final_traj