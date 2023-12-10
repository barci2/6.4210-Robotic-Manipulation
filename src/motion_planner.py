import time

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    LeafSystem,
    AbstractValue,
    DiagramBuilder,
    Trajectory,
    BsplineTrajectory,
    CompositeTrajectory,
    PiecewisePolynomial,
    PathParameterizedTrajectory,
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
    SpatialVelocity,
    JacobianWrtVariable
)

from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.utils import ConfigureParser
from pydrake.multibody import inverse_kinematics
from pydrake.solvers import SnoptSolver, IpoptSolver

from utils import ObjectTrajectory


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

        # used to figure out current gripper velocity
        iiwa_joint_vel = self.DeclareVectorInputPort(name="iiwa_current_vel", size=7)

        iiwa_joint_pos = self.DeclareVectorInputPort(name="iiwa_current_pos", size=7)
        

        self._traj_index = self.DeclareAbstractState(
            AbstractValue.Make(CompositeTrajectory([PiecewisePolynomial.FirstOrderHold(
                                                        [0, 1],
                                                        np.array([[0, 0]])
                                                    )]))
        )
        self.DeclareVectorOutputPort(
            "iiwa_position_command", 7, self.output_traj
        )
        # self._traj_wsg_index = self.DeclareAbstractState(
        #     AbstractValue.Make(PiecewisePolynomial())
        # )

        self.original_plant = original_plant
        self.meshcat = meshcat
        self.previous_compute_result = None  # BpslineTrajectory object

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
        NUM_STEPS = 75
        pos_3d_matrix = np.zeros((3,NUM_STEPS))
        ctr = 0
        for vis_t in np.linspace(traj_start_time, traj_end_time, NUM_STEPS):
            iiwa_pos = traj.value(vis_t)
            vis_plant.SetPositions(vis_plant_context, viz_iiwa, iiwa_pos)
            pos_3d = vis_plant.CalcRelativeTransform(vis_plant_context, vis_plant.world_frame(), vis_plant.GetFrameByName("iiwa_link_7")).translation()
            pos_3d_matrix[:,ctr] = pos_3d
            ctr += 1

        # Draw line
        self.meshcat.SetLine(name, pos_3d_matrix)


    def build_post_catch_trajectory(self, plant, plant_context, plant_autodiff, world_frame, gripper_frame, X_WCatch, catch_vel, iiwa_catch_pos, catch_time, traj_duration = 0.25):
        num_q = plant.num_positions()  # =7 (all of iiwa's joints)
        q0 = iiwa_catch_pos

        link_7_to_gripper_transform = RotationMatrix.MakeZRotation(np.pi / 2) @ RotationMatrix.MakeXRotation(np.pi / 2)

        trajopt = KinematicTrajectoryOptimization(num_q, 6)  # 6 control points in Bspline
        prog = trajopt.get_mutable_prog()
        self.setSolverSettings(prog)

        # Pick iiwa's catch position (in 7D) as initial guess for control points
        q_guess = np.tile(q0.reshape((7, 1)), (1, trajopt.num_control_points()))  # (7,4) np array
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

        trajopt.AddDurationConstraint(0, 1)

        # start constraint
        start_constraint = PositionConstraint(
            plant,
            world_frame,
            X_WCatch.translation(),  # upper limit
            X_WCatch.translation(),  # lower limit
            gripper_frame,
            [0, 0, 0.1],
            plant_context,
        )
        start_orientation_constraint = OrientationConstraint(
            plant,
            world_frame,
            X_WCatch.rotation(),  # orientation of gripper in world frame ...
            gripper_frame,
            link_7_to_gripper_transform,  # ... must equal origin in gripper frame
            0,  # theta bound
            plant_context
        )
        start_vel_constraint = SpatialVelocityConstraint(
            plant_autodiff,
            plant_autodiff.world_frame(),
            catch_vel,  # upper limit
            catch_vel,  # lower limit
            plant_autodiff.GetFrameByName("iiwa_link_7"),
            np.array([0, 0, 0.1]).reshape(-1,1),
            plant_autodiff.CreateDefaultContext(),
        )
        trajopt.AddPathPositionConstraint(start_constraint, 0)
        trajopt.AddPathPositionConstraint(start_orientation_constraint, 0)
        trajopt.AddVelocityConstraintAtNormalizedTime(start_vel_constraint, 0)
        prog.AddQuadraticErrorCost(
            np.eye(num_q), q0, trajopt.control_points()[:, 0]
        )

        # Calculate end position of end effector by simply integrating object's velocity starting from the catching pose
        X_WEnd = RigidTransform(X_WCatch.rotation(), X_WCatch.translation() + catch_vel.reshape((3,)) * traj_duration)
        # print(f"X_WEnd: {X_WEnd.translation()}")
        # print(f"X_WCatch: {X_WCatch.translation()}")

        # goal constraint
        goal_pos_constraint = PositionConstraint(
            plant,
            world_frame,
            X_WEnd.translation() - 0.25,  # upper limit
            X_WEnd.translation() + 0.25,  # lower limit
            gripper_frame,
            [0, 0, 0.1],
            plant_context,
        )
        # goal_orientation_constraint = OrientationConstraint(
        #     plant,
        #     world_frame,
        #     X_WEnd.rotation(),  # orientation of gripper in world frame ...
        #     gripper_frame,
        #     link_7_to_gripper_transform,  # ... must equal origin in gripper frame
        #     0.5,  # theta bound
        #     plant_context
        # )
        trajopt.AddPathPositionConstraint(goal_pos_constraint, 1)
        # trajopt.AddPathPositionConstraint(goal_orientation_constraint, 1)
        prog.AddQuadraticErrorCost(
            np.eye(num_q), q0, trajopt.control_points()[:, -1]
        )

        # End with zero velocity
        trajopt.AddPathVelocityConstraint(
            np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1
        )

        # Solve for trajectory
        solver = SnoptSolver()
        result = solver.Solve(prog)
        if not result.is_success():
            print("ERROR: post-catch traj opt solve failed: " + str(result.get_solver_id().name()))
            print(result.GetInfeasibleConstraintNames(prog))
        else:
            print("Post-catch traj opt solve succeeded.")
        traj = trajopt.ReconstructTrajectory(result)  # BSplineTrajectory

        # Time shift the trajectory
        time_shift = catch_time  # Time shift value in seconds
        time_scaling_trajectory = PiecewisePolynomial.FirstOrderHold(
            [time_shift, time_shift+traj_duration],  # Assuming two segments: initial and final times
            np.array([[0, 0]])  # Shifts start and end times by time_shift
        )
        time_shifted_traj = PathParameterizedTrajectory(
            traj, time_scaling_trajectory
        )

        return time_shifted_traj


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
                        acceptable_dur_err=0.0,
                        acceptable_pos_err=0.0,
                        theta_bound = 0.05,
                        acceptable_vel_err=0.5):
        
        trajopt.AddPathLengthCost(1.0)

        trajopt.AddPositionBounds(
            plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
        )
        trajopt.AddVelocityBounds(
            plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits()
        )

        trajopt.AddDurationConstraint(duration_target-acceptable_dur_err, duration_target+acceptable_dur_err)

        link_7_to_gripper_transform = RotationMatrix.MakeZRotation(np.pi / 2) @ RotationMatrix.MakeXRotation(np.pi / 2)

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
        trajopt.AddPathPositionConstraint(goal_pos_constraint, 1)
        trajopt.AddPathPositionConstraint(goal_orientation_constraint, 1)

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

        # end with velocity equal to object's velocity at that moment
        # DIVISION BY 3 IS TEMPORARY; HAVING SUCH HIGH ENDING VELOCITY MAKES IT VERY HARD FOR SNOPT TO SOLVE
        obj_vel_at_catch = obj_traj.EvalDerivative(obj_catch_t)[:3]/3  # (3,1) np array
        final_vel_constraint = SpatialVelocityConstraint(
            plant_autodiff,
            plant_autodiff.world_frame(),
            obj_vel_at_catch - acceptable_vel_err,  # upper limit
            obj_vel_at_catch + acceptable_vel_err,  # lower limit
            plant_autodiff.GetFrameByName("iiwa_link_7"),
            np.array([0, 0, 0.1]).reshape(-1,1),
            plant_autodiff.CreateDefaultContext(),
        )

        print(f"current_gripper_vel: {current_gripper_vel}")
        print(f"obj_vel_at_catch: {obj_vel_at_catch}")

        # collision constraints
        # collision_constraint = MinimumDistanceLowerBoundConstraint(
        #     plant, 0.001, plant_context, None, 0.01
        # )
        # evaluate_at_s = np.linspace(0, 1, 50)
        # for s in evaluate_at_s:
        #     trajopt.AddPathPositionConstraint(collision_constraint, s)

        return start_vel_constraint, final_vel_constraint
    

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

        # Get current iiwa positions
        q_current = self.get_input_port(4).Eval(context)  # "iiwa_current_pose" input port

        # Get current iiwa joint velocities from input port
        iiwa_vels = self.get_input_port(3).Eval(context)  # "iiwa_current_vel" input port
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
        # print(f"current_gripper_vel: {current_gripper_vel}")

        # Get selected grasp pose from input port
        grasp = self.get_input_port(0).Eval(context)
        X_WG = list(grasp.keys())[0]
        obj_catch_t = list(grasp.values())[0]
        if (X_WG.IsExactlyEqualTo(RigidTransform())):
            print("received default catch pose. returning from compute_traj.")
            return
        print(f"obj_catch_t: {obj_catch_t}")

        # If it's getting close to catch time, stop updating trajectory
        if obj_catch_t - context.get_time() < 0.3:
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
        AddMeshcatTriad(self.meshcat, "start", X_PT=X_WStart, opacity=0.5)
        self.meshcat.SetTransform("start", X_WStart)
        AddMeshcatTriad(self.meshcat, "goal", X_PT=X_WGoal, opacity=0.5)
        self.meshcat.SetTransform("goal", X_WGoal)

        num_q = plant.num_positions()  # =7 (all of iiwa's joints)
        q_nominal = np.array([0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0])  # nominal joint for joint-centering

        # If this is the very first traj opt (so we don't yet have a very good initial guess), do an interative optimization
        # if self.previous_compute_result is None:


        MAX_ITERATIONS = 8
        num_iter = 0
        cur_acceptable_duration_err=0.25
        cur_acceptable_pos_err=0.1
        cur_theta_bound=0.8
        cur_acceptable_vel_err=1.0
        final_traj = None
        while(num_iter < MAX_ITERATIONS):
            trajopt = KinematicTrajectoryOptimization(num_q, 8)  # 8 control points in Bspline
            prog = trajopt.get_mutable_prog()
            self.setSolverSettings(prog)
            
            if self.previous_compute_result is None and num_iter == 0:
                print("using ik for initial guess")
                # First solve the IK problem for X_WGoal. Then lin interp from start pos to goal pos,
                # use these points as control point initial guesses for the optimization.
                ik = inverse_kinematics.InverseKinematics(plant)
                q_variables = ik.q()  # Get variables for MathematicalProgram
                ik_prog = ik.prog()
                ik_prog.AddQuadraticErrorCost(np.identity(len(q_variables)), q_nominal, q_variables)
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
                ik_prog.SetInitialGuess(q_variables, q_nominal)
                ik_result = Solve(ik_prog)
                q_end = ik_result.GetSolution(q_variables)
                # Guess 8 control points in 7D for Bspline
                q_guess = np.linspace(q_current, q_end, 8).T  # (7,8) np array
                path_guess = BsplineTrajectory(trajopt.basis(), q_guess)
                trajopt.SetInitialGuess(path_guess)
            elif self.previous_compute_result is not None and num_iter == 0:
                print("using previous executed traj as initial guess")
                trajopt.SetInitialGuess(self.previous_compute_result)
            else:
                print("using previous iter as initial guess")
                trajopt.SetInitialGuess(final_traj)

            start_vel_constraint, final_vel_constraint = self.add_constraints(plant, 
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
            
            # For whatever reason, running AddVelocityConstraintAtNormalizedTime inside the function above causes segfault with no error message.
            trajopt.AddVelocityConstraintAtNormalizedTime(start_vel_constraint, 0)
            trajopt.AddVelocityConstraintAtNormalizedTime(final_vel_constraint, 1)

            # First solve with looser constraints
            solver = SnoptSolver()
            result = solver.Solve(prog)
            if not result.is_success():
                print(f"ERROR: num_iter={num_iter} Trajectory optimization failed: {result.get_solver_id().name()}")
                print(result.GetInfeasibleConstraintNames(prog))
                if final_traj is None:  # ensure final_traj is not None
                    final_traj = trajopt.ReconstructTrajectory(result)
                break
            else:
                print(f"num_iter={num_iter} Solve succeeded.")

            final_traj = trajopt.ReconstructTrajectory(result)  # BSplineTrajectory

            self.VisualizePath(final_traj, f"traj iter={num_iter}")

            # Make constraints more strict next iteration
            cur_acceptable_duration_err *= 0.75
            cur_acceptable_pos_err *= 0.75
            cur_theta_bound *= 0.75
            cur_acceptable_vel_err *= 0.75

            num_iter += 1
        

        # # Try again but with tighter constraints and using the last attempt as an initial guess
        # trajopt_refined = KinematicTrajectoryOptimization(num_q, 8)  # 8 control points in Bspline
        # prog_refined = trajopt_refined.get_mutable_prog()
        # self.setSolverSettings(prog_refined)
        # start_vel_constraint, final_vel_constraint = self.add_constraints(plant, 
        #                                                                   plant_context, 
        #                                                                   plant_autodiff, 
        #                                                                   trajopt_refined, 
        #                                                                   world_frame, 
        #                                                                   gripper_frame, 
        #                                                                   X_WStart, 
        #                                                                   X_WGoal,  
        #                                                                   obj_traj, 
        #                                                                   obj_catch_t,
        #                                                                   current_gripper_vel,
        #                                                                   duration_target=obj_catch_t-context.get_time()
        #                                                                   )
        
        # # For whatever reason, running AddVelocityConstraintAtNormalizedTime inside the function above causes segfault with no error message.
        # trajopt_refined.AddVelocityConstraintAtNormalizedTime(start_vel_constraint, 0)
        # trajopt_refined.AddVelocityConstraintAtNormalizedTime(final_vel_constraint, 1)

        # trajopt_refined.SetInitialGuess(solved_traj)

        # solver = SnoptSolver()
        # result = solver.Solve(prog)
        # if not result.is_success():
        #     print("ERROR: Second Trajectory optimization failed: " + str(result.get_solver_id().name()))
        #     print(result.GetInfeasibleConstraintNames(prog_refined))
        # else:
        #     print("Second solve succeeded.")

        # final_traj = trajopt_refined.ReconstructTrajectory(result)  # BSplineTrajectory

        # Shift trajectory in time so that it starts at the current time
        time_shift = context.get_time()  # Time shift value in seconds
        time_scaling_trajectory = PiecewisePolynomial.FirstOrderHold(
            [time_shift, time_shift+final_traj.end_time()],  # Assuming two segments: initial and final times
            np.array([[0, 1]])  # Shifts start and end times by time_shift
        )
        time_shifted_final_traj = PathParameterizedTrajectory(
            final_traj, time_scaling_trajectory
        )
        print(f"time_shifted_final_traj.start_time(): {time_shifted_final_traj.start_time()}")
        print(f"time_shifted_final_traj.end_time(): {time_shifted_final_traj.end_time()}")

        time_shifted_final_traj_end_time = time_shifted_final_traj.end_time()

        self.VisualizePath(time_shifted_final_traj, "final traj")

        state.get_mutable_abstract_state(int(self._traj_index)).set_value(time_shifted_final_traj)

        self.previous_compute_result = final_traj
        

        # obj_vel_at_catch = obj_traj.EvalDerivative(obj_catch_t)[:3]  # (3,1) np array
        # post_catch_traj = self.build_post_catch_trajectory(plant, 
        #                                                    plant_context, 
        #                                                    plant_autodiff, 
        #                                                    world_frame, 
        #                                                    gripper_frame, 
        #                                                    X_WGoal, 
        #                                                    obj_vel_at_catch, 
        #                                                    time_shifted_final_traj.value(time_shifted_final_traj_end_time), 
        #                                                    time_shifted_final_traj_end_time)

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
            output.SetFromVector(self.original_plant.GetPositions(self.original_plant.CreateDefaultContext(), self.original_plant.GetModelInstanceByName("iiwa")))

        else:
            # print("planner outputting iiwa position: " + str(traj_q.value(context.get_time())))
            output.SetFromVector(traj_q.value(context.get_time()))


    def output_wsg_traj(self, context, output):
        pass

        # grasp = self.get_input_port(0).Eval(context)
        # X_WG_Grasp = grasp.keys()[0]
        # obj_catch_t = grasp.values()[0]

        # X_WG = self.get_input_port(1).Eval(context)[self.original_plant.GetBodyByName("body", self.original_plant.GetModelInstanceByName("wsg")).index()]

        # position_diff = np.linalg.norm(X_WG_Grasp.translation() - X_WG.translation())
        # rotation_diff = X_WG_Grasp.rotation().angular_distance(X_WG.rotation())

        # # If robot is in catching position, close grippers
        # if position_diff < 0.005 and rotation_diff < 0.1:  # 5mm, 5 deg
        #     output.SetFromVector(np.array([0,0]))
        # else:
        #     output.SetFromVector(np.array([1,1]))