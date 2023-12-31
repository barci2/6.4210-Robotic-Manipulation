import time

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    LeafSystem,
    AbstractValue,
    DiagramBuilder,
    BsplineTrajectory,
    CompositeTrajectory,
    PiecewisePolynomial,
    PathParameterizedTrajectory,
    KinematicTrajectoryOptimization,
    Parser,
    PositionConstraint,
    OrientationConstraint,
    SpatialVelocityConstraint,
    RigidTransform,
    Solve,
    RotationMatrix,
    JacobianWrtVariable,
)

from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.utils import ConfigureParser
from pydrake.multibody import inverse_kinematics
from pydrake.solvers import SnoptSolver, IpoptSolver

from utils import ObjectTrajectory, calculate_obj_distance_to_gripper


class MotionPlanner(LeafSystem):
    """
    Perform Constrained Optimization to find optimal trajectory for iiwa to move
    to the grasping position.
    """

    def __init__(self, original_plant, meshcat):
        LeafSystem.__init__(self)

        grasp = AbstractValue.Make({RigidTransform(): 0})
        self.DeclareAbstractInputPort("grasp_selection", grasp)

        # used to figure out current gripper pose
        body_poses = AbstractValue.Make([RigidTransform()])
        self.DeclareAbstractInputPort("iiwa_current_pose", body_poses)

        obj_traj = AbstractValue.Make(ObjectTrajectory())
        self.DeclareAbstractInputPort("object_trajectory", obj_traj)

        iiwa_state = self.DeclareVectorInputPort(name="iiwa_state", size=14)  # 7 pos, 7 vel


        self._traj_index = self.DeclareAbstractState(
            AbstractValue.Make(CompositeTrajectory([PiecewisePolynomial.FirstOrderHold(
                                                        [0, 1],
                                                        np.array([[0, 0]])
                                                    )]))
        )

        # self._traj_wsg_index = self.DeclareAbstractState(
        #     AbstractValue.Make(CompositeTrajectory([PiecewisePolynomial.FirstOrderHold(
        #                                                 [0, 1],
        #                                                 np.array([[0, 0]])
        #                                             )]))
        # )

        self.DeclareVectorOutputPort(
            "iiwa_command", 14, self.output_traj  # 7 pos, 7 vel
        )

        self.DeclareVectorOutputPort(
            "iiwa_acceleration", 7, self.output_acceleration
        )

        self.DeclareVectorOutputPort(
            "wsg_command", 1, self.output_wsg_traj  # 7 pos, 7 vel
        )

        self.original_plant = original_plant
        self.meshcat = meshcat
        self.q_nominal = np.array([0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0])  # nominal joint for joint-centering
        self.q_end = None
        self.previous_compute_result = None  # BpslineTrajectory object
        self.visualize = True

        self.desired_wsg_state = 1  # default open

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.025, 0.0, self.compute_traj)


    def setSolverSettings(self, prog):
        prog.SetSolverOption(SnoptSolver().solver_id(), "Feasibility tolerance", 0.001)
        prog.SetSolverOption(SnoptSolver().solver_id(), "Major feasibility tolerance", 0.001)
        prog.SetSolverOption(SnoptSolver().solver_id(), "Minor feasibility tolerance", 0.001)
        prog.SetSolverOption(SnoptSolver().solver_id(), "Major optimality tolerance", 0.001)
        prog.SetSolverOption(SnoptSolver().solver_id(), "Minor optimality tolerance", 0.001)


    # Path Visualization
    def VisualizePath(self, traj, name):
        """
        Helper function that takes in trajopt basis and control points of Bspline
        and draws spline in meshcat.
        """
        # Build a new plant to do the forward kinematics to turn this Bspline into 3D coordinates
        builder = DiagramBuilder()
        vis_plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        viz_iiwa = Parser(vis_plant).AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision.urdf")[0]  # ModelInstance object
        vis_plant.WeldFrames(vis_plant.world_frame(), vis_plant.GetFrameByName("base"))
        vis_plant.Finalize()
        vis_plant_context = vis_plant.CreateDefaultContext()

        traj_start_time = traj.start_time()
        traj_end_time = traj.end_time()

        # Build matrix of 3d positions by doing forward kinematics at time steps in the bspline
        NUM_STEPS = 50
        pos_3d_matrix = np.zeros((3,NUM_STEPS))
        ctr = 0
        for vis_t in np.linspace(traj_start_time, traj_end_time, NUM_STEPS):
            iiwa_pos = traj.value(vis_t)
            vis_plant.SetPositions(vis_plant_context, viz_iiwa, iiwa_pos)
            pos_3d = vis_plant.CalcRelativeTransform(vis_plant_context, vis_plant.world_frame(), vis_plant.GetFrameByName("iiwa_link_7")).translation()
            pos_3d_matrix[:,ctr] = pos_3d
            ctr += 1

        # Draw line
        if self.visualize:
            self.meshcat.SetLine(name, pos_3d_matrix)


    def build_post_catch_trajectory(self,
                                    plant,
                                    world_frame,
                                    gripper_frame,
                                    X_WCatch,
                                    catch_vel,
                                    iiwa_catch_pos,
                                    catch_time,
                                    traj_duration=0.1
                                    ):

        # Assuming gripper continues at catch_vel for traj_duration time, figure out where iiwa ends up
        X_WEnd = RigidTransform(X_WCatch.rotation(), X_WCatch.translation() + catch_vel*traj_duration)

        print(f"X_WCatch: {X_WCatch}")
        print(f"X_WEnd: {X_WEnd}")

        link_7_to_gripper_transform = RotationMatrix.MakeZRotation(np.pi / 2) @ RotationMatrix.MakeXRotation(np.pi / 2)

        # Use IK to turn this into joint coords
        ik = inverse_kinematics.InverseKinematics(plant)
        q_variables = ik.q()  # Get variables for MathematicalProgram
        ik_prog = ik.prog()
        ik_prog.AddQuadraticErrorCost(np.identity(len(q_variables)), self.q_nominal, q_variables)
        ik.AddPositionConstraint(
            frameA=world_frame,
            frameB=gripper_frame,
            p_BQ=[0, 0, 0.1],
            p_AQ_lower=X_WEnd.translation(),
            p_AQ_upper=X_WEnd.translation(),
        )
        ik.AddOrientationConstraint(
            frameAbar=world_frame,
            R_AbarA=X_WEnd.rotation(),
            frameBbar=gripper_frame,
            R_BbarB=link_7_to_gripper_transform,
            theta_bound=0.05,
        )
        ik_prog.SetInitialGuess(q_variables, self.q_nominal)
        start = time.time()
        ik_result = Solve(ik_prog)
        print(f"ik time: {time.time()-start}")
        if not ik_result.is_success():
            print("ERROR: post-catch ik_result solve failed: " + str(ik_result.get_solver_id().name()))
            print(ik_result.GetInfeasibleConstraintNames(ik_prog))
        else:
            print("post-catch ik_result solve succeeded.")

        q_end = ik_result.GetSolution(q_variables)  # (7,) np array

        return q_end


        # complete_post_catch_traj = []

        # # Simple constant velocity trajectory in the direction of the object's velocity at catch time
        # NUM_WAYPOINTS = 10  # Number of waypoints for the trajectory
        # q_prev = iiwa_catch_pos.reshape((7,))  # (7,) np array
        # times = np.linspace(catch_time, catch_time+traj_duration, NUM_WAYPOINTS)
        # waypoints = np.linspace(X_WCatch.translation(), X_WEnd.translation(), NUM_WAYPOINTS)
        # for i, waypoint in enumerate(waypoints[1:], start=1):
        #     waypoint_pose = RigidTransform(X_WCatch.rotation(), waypoint)

        #     # Use IK to turn this into joint coords
        #     ik = inverse_kinematics.InverseKinematics(plant)
        #     q_variables = ik.q()  # Get variables for MathematicalProgram
        #     ik_prog = ik.prog()
        #     ik_prog.AddQuadraticErrorCost(np.identity(len(q_variables)), self.q_nominal, q_variables)
        #     ik.AddPositionConstraint(
        #         frameA=world_frame,
        #         frameB=gripper_frame,
        #         p_BQ=[0, 0, 0.1],
        #         p_AQ_lower=waypoint_pose.translation(),
        #         p_AQ_upper=waypoint_pose.translation(),
        #     )
        #     ik.AddOrientationConstraint(
        #         frameAbar=world_frame,
        #         R_AbarA=waypoint_pose.rotation(),
        #         frameBbar=gripper_frame,
        #         R_BbarB=RotationMatrix(),
        #         theta_bound=0.05,
        #     )
        #     ik_prog.SetInitialGuess(q_variables, self.q_nominal)
        #     start = time.time()
        #     ik_result = Solve(ik_prog)
        #     print(f"ik time: {time.time()-start}")
        #     if not ik_result.is_success():
        #         print("ERROR: post-catch ik_result solve failed: " + str(ik_result.get_solver_id().name()))
        #         print(ik_result.GetInfeasibleConstraintNames(ik_prog))
        #     else:
        #         print("post-catch ik_result solve succeeded.")

        #     q_waypoint = ik_result.GetSolution(q_variables)  # (7,) np array

        #     post_catch_traj = PiecewisePolynomial.FirstOrderHold([times[i-1], times[i]], np.column_stack([q_prev, q_waypoint]))

        #     q_prev = q_waypoint

        #     complete_post_catch_traj.append(post_catch_traj)

        # complete_post_catch_traj = CompositeTrajectory(complete_post_catch_traj)

        # return complete_post_catch_traj


    def add_constraints(self,
                        plant,
                        plant_context,
                        plant_autodiff,
                        trajopt,
                        world_frame,
                        gripper_frame,
                        X_WStart,
                        X_WGoal,
                        obj_traj,
                        obj_catch_t,
                        current_gripper_vel,
                        duration_target,
                        acceptable_dur_err=0.01,
                        acceptable_pos_err=0.02,
                        theta_bound = 0.1,
                        acceptable_vel_err=0.4):

        # trajopt.AddPathLengthCost(1.0)

        trajopt.AddPositionBounds(
            plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
        )
        trajopt.AddVelocityBounds(
            plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits()
        )

        obj_vel_at_catch = obj_traj.EvalDerivative(obj_catch_t)[:3]
        # if object is up to 0.1m radius, make gripper arrive at catching pose that distance early
        # gripper_early_time = 0.08 / np.linalg.norm(obj_vel_at_catch)
        # print(f"gripper_early_time: {gripper_early_time}")
        # duration_target -= gripper_early_time

        pre_catch_time_offset = 0.6
        catch_time_normalized = duration_target / (duration_target + pre_catch_time_offset)
        duration_target += pre_catch_time_offset

        pre_gripper_time = duration_target - pre_catch_time_offset - 0.12
        pre_gripper_normalized_time = pre_gripper_time / duration_target
        X_WO = obj_traj.value(obj_catch_t)
        X_OG_W = X_WO.inverse() @ X_WGoal
        X_WPreGoal = obj_traj.value(obj_catch_t - 0.12) @ X_OG_W

        obj_vel_at_pre_catch = obj_traj.EvalDerivative(obj_catch_t - 0.12)[:3]

        # print(f"duration_target: {duration_target}")
        trajopt.AddDurationConstraint(duration_target-acceptable_dur_err, duration_target+acceptable_dur_err)

        link_7_to_gripper_transform = RotationMatrix.MakeZRotation(np.pi / 2) @ RotationMatrix.MakeXRotation(np.pi / 2)

        # To avoid segfaulting on the constraints being deallocated
        global start_pos_constraint, start_orientation_constraint,\
               pre_goal_pos_constraint, pre_goal_orientation_constraint, pre_goal_vel_constraint,\
               goal_pos_constraint, goal_orientation_constraint, \
               start_vel_constraint, final_vel_constraint

        # start constraint
        start_pos_constraint = PositionConstraint(
            plant,
            world_frame,
            X_WStart.translation() - acceptable_pos_err,  # lower limit
            X_WStart.translation() + acceptable_pos_err,  # upper limit
            gripper_frame,
            [0, 0, 0.1],
            plant_context,
        )
        start_orientation_constraint = OrientationConstraint(
            plant,
            world_frame,
            X_WStart.rotation(),  # orientation of X_WStart in world frame ...
            gripper_frame,
            link_7_to_gripper_transform,  # ... must equal origin in gripper frame
            theta_bound,
            plant_context
        )
        trajopt.AddPathPositionConstraint(start_pos_constraint, 0)
        trajopt.AddPathPositionConstraint(start_orientation_constraint, 0)

        # Pre goal constraint
        pre_goal_pos_constraint = PositionConstraint(
            plant,
            world_frame,
            X_WPreGoal.translation() - max(acceptable_pos_err * 25, 0.2),  # lower limit
            X_WPreGoal.translation() + max(acceptable_pos_err * 25, 0.2),  # upper limit
            gripper_frame,
            [0, 0, 0.1],
            plant_context
        )
        pre_goal_orientation_constraint = OrientationConstraint(
            plant,
            world_frame,
            X_WPreGoal.rotation(),  # orientation of X_WGoal in world frame ...
            gripper_frame,
            link_7_to_gripper_transform,  # ... must equal origin in gripper frame
            theta_bound,
            plant_context
        )
        trajopt.AddPathPositionConstraint(pre_goal_pos_constraint, pre_gripper_normalized_time)
        trajopt.AddPathPositionConstraint(pre_goal_orientation_constraint, pre_gripper_normalized_time)

        # goal constraint
        goal_pos_constraint = PositionConstraint(
            plant,
            world_frame,
            X_WGoal.translation() - acceptable_pos_err,  # lower limit
            X_WGoal.translation() + acceptable_pos_err,  # upper limit
            gripper_frame,
            [0, 0, 0.1],
            plant_context,
        )
        goal_orientation_constraint = OrientationConstraint(
            plant,
            world_frame,
            X_WGoal.rotation(),  # orientation of X_WGoal in world frame ...
            gripper_frame,
            link_7_to_gripper_transform,  # ... must equal origin in gripper frame
            theta_bound,
            plant_context
        )
        # print(f"catch_time_normalized: {catch_time_normalized}")
        trajopt.AddPathPositionConstraint(goal_pos_constraint, catch_time_normalized)
        trajopt.AddPathPositionConstraint(goal_orientation_constraint, catch_time_normalized)

        # Start with velocity equal to iiwa's current velocity
        # Current limitation: SpatialVelocityConstraint only takes into account translational velocity; not rotational
        start_vel_constraint = SpatialVelocityConstraint(
            plant_autodiff,
            plant_autodiff.world_frame(),
            current_gripper_vel - acceptable_vel_err,  # upper limit
            current_gripper_vel + acceptable_vel_err,  # lower limit
            plant_autodiff.GetFrameByName("iiwa_link_7"),
            np.array([0, 0, 0.1]).reshape(-1,1),
            plant_autodiff.CreateDefaultContext(),
        )

        # end with same directional velocity as object at catch
        catch_vel = obj_vel_at_catch * 0.3 # (3,1) np array
        final_vel_constraint = SpatialVelocityConstraint(
            plant_autodiff,
            plant_autodiff.world_frame(),
            catch_vel - acceptable_vel_err,  # upper limit
            catch_vel + acceptable_vel_err,  # lower limit
            plant_autodiff.GetFrameByName("iiwa_link_7"),
            np.array([0, 0, 0.1]).reshape(-1,1),
            plant_autodiff.CreateDefaultContext(),
        )

        trajopt.AddVelocityConstraintAtNormalizedTime(start_vel_constraint, 0)
        trajopt.AddVelocityConstraintAtNormalizedTime(final_vel_constraint, catch_time_normalized)


    def compute_traj(self, context, state):
        print("motion_planner update event")

        # if self.previous_compute_result != None:
        #     return

        obj_traj = self.get_input_port(2).Eval(context)
        if (obj_traj == ObjectTrajectory()):  # default output of TrajectoryPredictor system; means that it hasn't seen the object yet
            # print("received default obj traj (in compute_traj). returning from compute_traj.")
            return

        # Get current gripper pose from input port
        body_poses = self.get_input_port(1).Eval(context)  # "iiwa_current_pose" input port
        gripper_body_idx = self.original_plant.GetBodyByName("body").index()  # BodyIndex object
        current_gripper_pose = body_poses[gripper_body_idx]  # RigidTransform object

        # Get current iiwa positions and velocities
        iiwa_state = self.get_input_port(3).Eval(context)
        q_current = iiwa_state[:7]
        iiwa_vels = iiwa_state[7:]

        # Build a new plant to do calculate the velocity Jacobian
        builder = DiagramBuilder()
        j_plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        j_iiwa = Parser(j_plant).AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision.urdf")[0]  # ModelInstance object
        j_plant.WeldFrames(j_plant.world_frame(), j_plant.GetFrameByName("base"))
        j_plant.Finalize()
        j_plant_context = j_plant.CreateDefaultContext()
        j_plant.SetPositions(j_plant_context, j_iiwa, q_current)
        # Build Jacobian to solve for translational velocity from joint velocities
        J = j_plant.CalcJacobianTranslationalVelocity(j_plant_context,
                                                      JacobianWrtVariable.kQDot,
                                                      j_plant.GetFrameByName("iiwa_link_7"),
                                                      [0, 0, 0.1],  # offset from iiwa_link_7_ to where gripper would be
                                                      j_plant.world_frame(),  #
                                                      j_plant.world_frame()  # frame that translational velocity should be expressed in
                                                      )
        current_gripper_vel = np.dot(J, iiwa_vels)

        # Get selected grasp pose from input port
        grasp = self.get_input_port(0).Eval(context)
        X_WG = list(grasp.keys())[0]
        obj_catch_t = list(grasp.values())[0]
        if (X_WG.IsExactlyEqualTo(RigidTransform())):
            print("received default catch pose. returning from compute_traj.")
            return
        # print(f"obj_catch_t: {obj_catch_t}")

        # If it's getting close to catch time, stop updating trajectory
        if obj_catch_t - context.get_time() < 0.2:
            return

        # Setup a new MBP with just the iiwa which the KinematicTrajectoryOptimization will use
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        iiwa = Parser(plant).AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision.urdf")[0]  # ModelInstance object
        world_frame = plant.world_frame()
        base_frame = plant.GetFrameByName("base")
        gripper_frame = plant.GetFrameByName("iiwa_link_7")
        plant.WeldFrames(world_frame, base_frame)  # Weld iiwa to world
        plant.Finalize()
        plant_context = plant.CreateDefaultContext()

        # Create auto-differentiable version the plant in order to set velocity constraints
        plant_autodiff = plant.ToAutoDiffXd()

        X_WStart = current_gripper_pose
        X_WGoal = X_WG
        # print(f"X_WStart: {X_WStart}")
        # print(f"X_WGoal: {X_WGoal}")
        if self.visualize:
            # AddMeshcatTriad(self.meshcat, "start", X_PT=X_WStart, opacity=0.5)
            # self.meshcat.SetTransform("start", X_WStart)
            AddMeshcatTriad(self.meshcat, "goal", X_PT=X_WGoal, opacity=0.5)
            self.meshcat.SetTransform("goal", X_WGoal)

        obj_vel_at_catch = obj_traj.EvalDerivative(obj_catch_t)[:3]  # (3,1) np array
        print(f"current_gripper_vel: {current_gripper_vel}")
        print(f"obj_vel_at_catch: {obj_vel_at_catch}")

        num_q = plant.num_positions()  # =7 (all of iiwa's joints)

        # If this is the very first traj opt (so we don't yet have a very good initial guess), do an interative optimization
        MAX_ITERATIONS = 12
        if self.previous_compute_result is None:
            num_iter = 0
            cur_acceptable_duration_err=0.05
            cur_acceptable_pos_err=0.1
            cur_theta_bound=0.8
            cur_acceptable_vel_err=2.0
            final_traj = None
            while(num_iter < MAX_ITERATIONS):
                trajopt = KinematicTrajectoryOptimization(num_q, 8)  # 8 control points in Bspline
                prog = trajopt.get_mutable_prog()
                self.setSolverSettings(prog)

                if num_iter == 0:
                    print("using ik for initial guess")
                    # First solve the IK problem for X_WGoal. Then lin interp from start pos to goal pos,
                    # use these points as control point initial guesses for the optimization.
                    ik = inverse_kinematics.InverseKinematics(plant)
                    q_variables = ik.q()  # Get variables for MathematicalProgram
                    ik_prog = ik.prog()
                    ik_prog.AddQuadraticErrorCost(np.identity(len(q_variables)), self.q_nominal, q_variables)
                    ik.AddPositionConstraint(
                        frameA=world_frame,
                        frameB=gripper_frame,
                        p_BQ=[0, 0, 0.1],
                        p_AQ_lower=X_WGoal.translation(),
                        p_AQ_upper=X_WGoal.translation(),
                    )
                    ik.AddOrientationConstraint(
                        frameAbar=world_frame,
                        R_AbarA=X_WGoal.rotation(),
                        frameBbar=gripper_frame,
                        R_BbarB=RotationMatrix(),
                        theta_bound=0.05,
                    )
                    ik_prog.SetInitialGuess(q_variables, self.q_nominal)
                    ik_result = Solve(ik_prog)
                    if not ik_result.is_success():
                        print("ERROR: ik_result solve failed: " + str(ik_result.get_solver_id().name()))
                        print(ik_result.GetInfeasibleConstraintNames(ik_prog))
                    else:
                        print("ik_result solve succeeded.")

                    q_end = ik_result.GetSolution(q_variables)
                    # Guess 8 control points in 7D for Bspline
                    q_guess = np.linspace(q_current, q_end, 8).T  # (7,8) np array
                    path_guess = BsplineTrajectory(trajopt.basis(), q_guess)
                    trajopt.SetInitialGuess(path_guess)
                else:
                    print("using previous iter as initial guess")
                    trajopt.SetInitialGuess(final_traj)

                self.add_constraints(plant,
                                     plant_context,
                                     plant_autodiff,
                                     trajopt,
                                     world_frame,
                                     gripper_frame,
                                     X_WStart,
                                     X_WGoal,
                                     obj_traj,
                                     obj_catch_t,
                                     current_gripper_vel,
                                     duration_target=obj_catch_t-context.get_time(),
                                     acceptable_dur_err=cur_acceptable_duration_err,
                                     acceptable_pos_err=cur_acceptable_pos_err,
                                     theta_bound=cur_theta_bound,
                                     acceptable_vel_err=cur_acceptable_vel_err)
                
                # First solve with looser constraints
                solver = SnoptSolver()
                result = solver.Solve(prog)
                if not result.is_success():
                    print(f"ERROR: num_iter={num_iter} Trajectory optimization failed: {result.get_solver_id().name()}")
                    print(result.GetInfeasibleConstraintNames(prog))
                    if final_traj is None:  # ensure final_traj is not None
                        final_traj = trajopt.ReconstructTrajectory(result)
                    # raise RuntimeError()
                    break
                else:
                    print(f"num_iter={num_iter} Solve succeeded.")

                final_traj = trajopt.ReconstructTrajectory(result)  # BSplineTrajectory

                self.VisualizePath(final_traj, f"traj iter={num_iter}")

                # Make constraints more strict next iteration
                cur_acceptable_duration_err *= 0.875
                cur_acceptable_pos_err *= 0.875
                cur_theta_bound *= 0.875
                cur_acceptable_vel_err *= 0.875

                num_iter += 1

        # If this is not the first cycle (so we have a good initial guess already), then just go straight to an optimization w/strict constraints
        else:
            # Undraw previous trajctories that aren't actually being followed
            for i in range(MAX_ITERATIONS):
                try:
                    if self.visualize:
                        self.meshcat.Delete(f"traj iter={i}")
                except:
                    pass

            print("using previous cycle's executed trajectory as initial guess")
            trajopt = KinematicTrajectoryOptimization(num_q, 8)  # 8 control points in Bspline
            prog = trajopt.get_mutable_prog()
            self.setSolverSettings(prog)
            self.add_constraints(plant,
                                 plant_context,
                                 plant_autodiff,
                                 trajopt,
                                 world_frame,
                                 gripper_frame,
                                 X_WStart,
                                 X_WGoal,
                                 obj_traj,
                                 obj_catch_t,
                                 current_gripper_vel,
                                 duration_target=obj_catch_t-context.get_time())

            trajopt.SetInitialGuess(self.previous_compute_result)

            solver = SnoptSolver()
            result = solver.Solve(prog)
            if not result.is_success():
                print("ERROR: Tight Trajectory optimization failed: " + str(result.get_solver_id().name()))
                print(result.GetInfeasibleConstraintNames(prog))
            else:
                print("Tight solve succeeded.")

            final_traj = trajopt.ReconstructTrajectory(result)  # BSplineTrajectory

        # Shift trajectory in time so that it starts at the current time
        time_shift = context.get_time()  # Time shift value in seconds
        time_scaling_traj = PiecewisePolynomial.FirstOrderHold(
            [time_shift, time_shift+final_traj.end_time()],  # Assuming two segments: initial and final times
            np.array([[0, final_traj.end_time()-final_traj.start_time()]])  # Shifts start and end times by time_shift
        )
        time_shifted_final_traj = PathParameterizedTrajectory(
            final_traj, time_scaling_traj
        )
        # print(f"time_shifted_final_traj.start_time(): {time_shifted_final_traj.start_time()}")
        # print(f"time_shifted_final_traj.end_time(): {time_shifted_final_traj.end_time()}")

        self.VisualizePath(time_shifted_final_traj, "final traj")

        state.get_mutable_abstract_state(int(self._traj_index)).set_value(time_shifted_final_traj)

        self.previous_compute_result = final_traj  # save the solved trajectory to use as initial guess next iteration

        self.obj_vel_at_catch = obj_traj.EvalDerivative(obj_catch_t)[:3]

        # self.q_end = self.build_post_catch_trajectory(plant,
        #                                         world_frame,
        #                                         gripper_frame,
        #                                         X_WGoal,
        #                                         obj_vel_at_catch,
        #                                         time_shifted_final_traj.value(time_shifted_final_traj.end_time()),
        #                                         time_shifted_final_traj.end_time())

        # post_catch_traj = self.build_post_catch_trajectory(plant,
        #                                                    world_frame,
        #                                                    gripper_frame,
        #                                                    X_WGoal,
        #                                                    obj_vel_at_catch,
        #                                                    time_shifted_final_traj.value(time_shifted_final_traj.end_time()),
        #                                                    time_shifted_final_traj.end_time())

        # complete_traj = CompositeTrajectory([time_shifted_final_traj, post_catch_traj])

        # self.VisualizePath(complete_traj, "complete traj")

        # state.get_mutable_abstract_state(int(self._traj_index)).set_value(complete_traj)

    def output_traj(self, context, output):
        # Just set value at output port according to context time and trajectory state variable
        traj_q = context.get_mutable_abstract_state(int(self._traj_index)).get_value()

        # traj_q.rows() == 1 basically means traj_q is the default;
        # either object trajectory hasn't finished predicting yet, or grasp hasn't been selected yet,
        if (traj_q.rows() == 1):
            # print("planner outputting default iiwa position")
            output.SetFromVector(np.append(
                self.original_plant.GetPositions(self.original_plant.CreateDefaultContext(), self.original_plant.GetModelInstanceByName("iiwa")),
                np.zeros((7,))
            ))

        else:
            # if context.get_time() <= traj_q.end_time():
            #     # print("planner outputting iiwa position: " + str(traj_q.value(context.get_time())))
            #     output.SetFromVector(np.append(
            #         traj_q.value(context.get_time()),
            #         traj_q.EvalDerivative(context.get_time())
            #     ))
            # else:  # return the ik result computed at end position
            #     if self.q_end is not None:
            #         output.SetFromVector(np.append(self.q_end, np.zeros(7)))

            output.SetFromVector(np.append(
                traj_q.value(context.get_time()),
                traj_q.EvalDerivative(context.get_time())
            ))


    def output_acceleration(self, context, output):
        traj_q = context.get_mutable_abstract_state(int(self._traj_index)).get_value()

        if (traj_q.rows() == 1):
            # print("planner outputting default 0 acceleration")
            output.SetFromVector(np.zeros((7,)))
        else:
            output.SetFromVector(traj_q.EvalDerivative(context.get_time(), 2))


    def output_wsg_traj(self, context, output):
        # Get current gripper pose from input port
        body_poses = self.get_input_port(1).Eval(context)  # "iiwa_current_pose" input port
        gripper_body_idx = self.original_plant.GetBodyByName("body").index()  # BodyIndex object
        current_gripper_pose = body_poses[gripper_body_idx]  # RigidTransform object

        # find body index of obj being thrown
        if self.original_plant.HasBodyNamed("Tennis_ball"):
            obj_body_name = "Tennis_ball"
        elif self.original_plant.HasBodyNamed("Banana"):
            obj_body_name = "Banana"
        elif self.original_plant.HasBodyNamed("pill_bottle"):
            obj_body_name = "pill_bottle"

        # Get current object pose from input port
        obj_body_idx = self.original_plant.GetBodyByName(obj_body_name).index()  # BodyIndex object
        current_obj_pose = body_poses[obj_body_idx]  # RigidTransform object

        obj_distance_to_grasp, vector_gripper_to_obj = calculate_obj_distance_to_gripper(current_gripper_pose, current_obj_pose)

        DISTANCE_TRESHOLD_TO_GRASP = 0.01

        # first comparison measures distance in general (to ensure obj is roughly in range for a catch); 
        # second comparison is more precise, measures obj distance to grasp in gripper frame z-axis
        if (np.linalg.norm(vector_gripper_to_obj) < 0.25 and obj_distance_to_grasp < DISTANCE_TRESHOLD_TO_GRASP) or self.desired_wsg_state == 0:
            output.SetFromVector(np.array([0]))  # closed
            self.desired_wsg_state = 0  # grippers should be closed from now on
        else:
            output.SetFromVector(np.array([1]))  # open