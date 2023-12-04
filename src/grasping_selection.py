from pydrake.all import (
    AbstractValue,
    Concatenate,
    Trajectory,
    LeafSystem,
    PointCloud,
    AddMultibodyPlantSceneGraph,
    Box,
    DiagramBuilder,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Quaternion,
    InverseKinematics,
    Solve
)
from manipulation import running_as_notebook
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.mustard_depth_camera_example import MustardPointCloud
from manipulation.scenarios import AddMultibodyTriad
from manipulation.utils import ConfigureParser, LoadDataResource

import time
import numpy as np
from scipy.spatial import KDTree


class TrajectoryWrapper():
    def __init__(self):
        # Bare constructor that is used when defining the type of the AbstractInputPort
        self.trajectory = None

    def add_trajectory_object(self, trajectory):
        # trajectory is a drake Trajectory object
        self.trajectory = trajectory

    def value(self, t):
        return self.trajectory.value(t)

class GraspSelector(LeafSystem):
    """
    Use method described in this paper: https://arxiv.org/pdf/1706.09911.pdf to
    sample potential grasps until finding one at a desirable position for iiwa.
    """

    def __init__(self, plant, scene_graph, meshcat):
        LeafSystem.__init__(self)

        obj_pc = AbstractValue.Make(PointCloud())
        obj_traj = AbstractValue.Make(TrajectoryWrapper())
        self.DeclareAbstractInputPort("object_pc", obj_pc)
        self.DeclareAbstractInputPort("object_trajectory", obj_traj)
        
        port = self.DeclareAbstractOutputPort(
            "grasp_selection",
            lambda: AbstractValue.Make({RigidTransform(): 0}),  # dict mapping grasp to a grasp time
            self.SelectGrasp,
        )
        port.disable_caching_by_default()

        self._rng = np.random.default_rng()
        self.plant = plant
        self.scene_graph = scene_graph
        self.meshcat = meshcat


    def draw_grasp_candidate(self, X_G, prefix="gripper", draw_frames=True, refresh=False):
        """
        Helper function to visualize grasp.
        """
        if refresh:
            self.meshcat.Delete()
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        parser = Parser(plant)
        ConfigureParser(parser)
        gripper = parser.AddModelsFromUrl(
            "package://manipulation/schunk_wsg_50_welded_fingers.sdf"
        )
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body"), X_G)
        if draw_frames:
            AddMultibodyTriad(plant.GetFrameByName("body"), scene_graph)
        plant.Finalize()

        params = MeshcatVisualizerParams()
        params.prefix = prefix
        meshcat_vis = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, self.meshcat, params
        )

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        diagram.ForcedPublish(context)

    def check_collision(self, obj_pc, X_G):
        """
        TODO: Speed up this function.
        """
        return False
        # plant_context = self.plant.GetMyContextFromRoot(self.context)
        # scene_graph_context = self.scene_graph.GetMyContextFromRoot(self.context)
        # self.plant.SetFreeBodyPose(plant_context, self.plant.GetBodyByName("body"), X_G)

        # query_object = self.scene_graph.get_query_output_port().Eval(
        #     scene_graph_context
        # )

        # for pt in obj_pc.xyzs().T:
        #     distances = query_object.ComputeSignedDistanceToPoint(pt)
        #     for body_index in range(len(distances)):
        #         distance = distances[body_index].distance
        #         if distance < 0:
        #             return True  # Collision

        # return False


    def compute_darboux_frame(self, index, obj_pc, kdtree, ball_radius=0.002, max_nn=50):
        """
        Given a index of the pointcloud, return a RigidTransform from world to the
        Darboux frame at that point.

        Args:
        - index (int): index of the pointcloud.
        - obj_pc (PointCloud object): pointcloud of the object.
        - kdtree (scipy.spatial.KDTree object): kd tree to use for nn search.
        - ball_radius (float): ball_radius used for nearest-neighbors search
        - max_nn (int): maximum number of points considered in nearest-neighbors search.
        """
        points = obj_pc.xyzs()  # 3xN np array of points
        normals = obj_pc.normals()  # 3xN np array of normals

        # 1. Find nearest neighbors to point in PC
        nn_distances, nn_indices = kdtree.query(points[:,index], max_nn, distance_upper_bound=ball_radius)
        finite_indices = np.isfinite(nn_distances)
        nn_indices = nn_indices[finite_indices]

        # 2. compute N, covariance matrix of all normal vectors in neighborhood
        nn_normals = normals[:, nn_indices]  # 3xK matrix where K is the number of neighbors the point has
        N = nn_normals @ nn_normals.T  # 3x3

        # 3. Eigen decomp (v1 = normal, v2 = major tangent, v3 = minor tangent)
        # The Eigenvectors create an orthogonal basis (note that N is symmetric) that can be used to construct a rotation matrix
        eig_vals, eig_vecs = np.linalg.eig(N)  # vertically stacked eig vecs
        # Sort the eigenvectors based on the eigenvalues
        sorted_indices = np.argsort(eig_vals)[::-1]  # Get the indices that would sort the eigenvalues in descending order
        eig_vals = eig_vals[sorted_indices]  # Sort the eigenvalues
        eig_vecs = eig_vecs[:, sorted_indices]  # Sort the eigenvectors accordingly

        # 4. Ensure v1 (eig vec corresponding to largest eig val) points into object
        if (eig_vecs[:,0] @ normals[:,index] > 0):  # if dot product with normal is pos, that means v1 is pointing out
            eig_vecs[:,0] *= -1  # flip v1

        # 5. Construct Rotation matrix to X_WF (by horizontal stacking v2 v1 v3)
        # This works bc rotation matrices are, by definition, 3 horizontally stacked orthonormal columns
        # Also, we choose the order [v2 v1 v3] bc v1 (with largest eigen value) corresponds to y-axis, v2 (with 2nd largest eigen value) corresponds to major axis of curvature (x-axis), and v3 (smallest eignvalue) correponds to minor axis of curvature (z-axis)
        R = np.hstack((eig_vecs[:,1:2], eig_vecs[:,0:1], eig_vecs[:,2:3]))  # need to reshape vectors to col vectors
        # print(R)

        # 6. Check if matrix is improper (is actually both a rotation and reflection), if so, fix it
        if np.linalg.det(R) < 0:  # if det is neg, this means rot matrix is improper
            R[:, 0] *= -1  # multiply left column (v2) by -1 to fix improperness

        #7. Create a rigid transform with the rotation of the normal and position of the point in the PC
        X_WF = RigidTransform(RotationMatrix(R), points[:,index])  # modify here.

        return X_WF



    def check_nonempty(self, obj_pc, X_WG, visualize=False):
        """
        Check if the "closing region" of the gripper is nonempty by transforming the
        pointclouds to gripper coordinates.

        Args:
        - obj_pc (PointCloud object): pointcloud of the object.
        - X_WG (Drake RigidTransform): transform of the gripper.
        Return:
        - is_nonempty (boolean): boolean set to True if there is a point within
            the cropped region.
        """
        obj_pc_W_np = obj_pc.xyzs()

        # Bounding box of the closing region written in the coordinate frame of the gripper body.
        # Do not modify
        crop_min = [-0.05, 0.05, -0.00625]
        crop_max = [0.05, 0.1125, 0.00625]

        # Transform the pointcloud to gripper frame.
        X_GW = X_WG.inverse()
        obj_pc_G_np = X_GW.multiply(obj_pc_W_np)

        # Check if there are any points within the cropped region.
        indices = np.all(
            (
                crop_min[0] <= obj_pc_G_np[0, :],
                obj_pc_G_np[0, :] <= crop_max[0],
                crop_min[1] <= obj_pc_G_np[1, :],
                obj_pc_G_np[1, :] <= crop_max[1],
                crop_min[2] <= obj_pc_G_np[2, :],
                obj_pc_G_np[2, :] <= crop_max[2],
            ),
            axis=0,
        )

        is_nonempty = indices.any()

        if visualize:
            self.meshcat.Delete()
            obj_pc_G = PointCloud(obj_pc)
            obj_pc_G.mutable_xyzs()[:] = obj_pc_G_np

            self.draw_grasp_candidate(RigidTransform())
            self.meshcat.SetObject("cloud", obj_pc_G)

            box_length = np.array(crop_max) - np.array(crop_min)
            box_center = (np.array(crop_max) + np.array(crop_min)) / 2.0
            self.meshcat.SetObject(
                "closing_region",
                Box(box_length[0], box_length[1], box_length[2]),
                Rgba(1, 0, 0, 0.3),
            )
            self.meshcat.SetTransform("closing_region", RigidTransform(box_center))

        return is_nonempty



    def compute_candidate_grasps(self, obj_pc, candidate_num=10, random_seed=5):
        """
        Args:
            - obj_pc (PointCloud object): pointcloud of the object.
            - candidate_num (int) : number of desired candidates.
            - random_seed (int) : seed for rng, used for grading.
        Return:
            - candidate_lst (list of drake RigidTransforms) : candidate list of grasps.
        """

        # Constants for random variation
        x_min = -0.03
        x_max = 0.03
        phi_min = -np.pi / 3
        phi_max = np.pi / 3
        # np.random.seed(random_seed)

        # Build KD tree for the pointcloud.
        kdtree = KDTree(obj_pc.xyzs().T)
        ball_radius = 0.002

        candidate_lst = []  # list of candidates, given by RigidTransforms.

        def compute_candidate(idx, obj_pc, kdtree, ball_radius, x_min, x_max, phi_min, phi_max, candidate_lst_lock, candidate_lst):
            X_WF = self.compute_darboux_frame(idx, obj_pc, kdtree, ball_radius)  # find Darboux frame of random point

            # Compute random x-translation and z-rotation to modify gripper pose
            # random_x = np.random.uniform(x_min, x_max)
            # random_z_rot = np.random.uniform(phi_min, phi_max)
            # X_WG = X_WF @ RigidTransform(RotationMatrix.MakeZRotation(random_z_rot), np.array([random_x, 0, 0]))

            new_X_WG = X_WF @ RigidTransform(np.array([0, -0.05, 0]))  # Move grasper back by fixed amount

            # check_collision takes most of the runtime
            if (self.check_collision(obj_pc, new_X_WG) is not True) and self.check_nonempty(obj_pc, new_X_WG):  # no collision, and there is an object between fingers
                with candidate_lst_lock:
                    candidate_lst.append(new_X_WG)

        import threading

        threads = []
        candidate_lst_lock = threading.Lock()
        for i in range(candidate_num):
            random_idx = np.random.randint(0, obj_pc.size())
            t = threading.Thread(target=compute_candidate, args=(random_idx, 
                                                                obj_pc, 
                                                                kdtree, 
                                                                ball_radius, 
                                                                x_min, 
                                                                x_max, 
                                                                phi_min, 
                                                                phi_max, 
                                                                candidate_lst_lock, 
                                                                candidate_lst))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return candidate_lst


    def grasp_cost(self, obj_pc_centroid, X_OG, t):
        """
        Defines cost function that is used to pick best grasp sample.

        Args:
            obj_pc_centroid: (3,) np array
            X_OG: RigidTransform containing gripper pose for this grasp in obj frame
            t: float, time at which the grasp occurs
        """

        # Compute distance from Y-axis ray of gripper frame to objects' centroid.
        obj_pc_centroid_relative_to_X_OG = obj_pc_centroid - X_OG.translation()
        X_OG_y_axis_vector = X_OG.rotation().matrix()[:, 1]
        projection_obj_pc_centroid_relative_to_X_OG_onto_X_OG_y_axis_vector = (np.dot(obj_pc_centroid_relative_to_X_OG, X_OG_y_axis_vector) / np.linalg.norm(X_OG_y_axis_vector)) * X_OG_y_axis_vector  # Equation for projection of one vector onto another
        # On the order of 0 - 0.05
        distance_obj_pc_centroid_to_X_OG_y_axis = np.linalg.norm(obj_pc_centroid_relative_to_X_OG - projection_obj_pc_centroid_relative_to_X_OG_onto_X_OG_y_axis_vector)

        # Transform the grasp pose from object frame to world frame
        obj_pose_data = self.obj_traj.value(t)  # (7,1) vector containing x,y,z,q0,q1,q2,q3
        obj_pose_quaterion = obj_pose_data[3:]
        obj_pose_position = obj_pose_data[:3]
        X_WO = RigidTransform(Quaternion(obj_pose_quaterion), obj_pose_position)
        X_WG = X_WO @ X_OG

        # Add cost associated with whether X_WG's y-axis points away from iiwa (which is what we want)
        X_WG_to_z_axis_vector = np.append(X_WG.translation()[:2], 0)  # basically replacing z with 0
        X_WG_y_axis_vector = X_WO @ X_OG_y_axis_vector
        # On the order of 0 - PI
        angle = np.arccos(np.dot(X_WG_to_z_axis_vector, X_WG_y_axis_vector) / (np.linalg.norm(X_WG_to_z_axis_vector) * np.linalg.norm(X_WG_y_axis_vector)))

        # Add cost associated with whether object is able to fly in between two fingers of gripper
        # Z-axis of gripper should be aligned with derivative of obj trajectory


        # Use IK to find joint positions for the given grasp pose
        # ik = InverseKinematics(self.plant)
        # ik.AddPositionConstraint(...)
        # ik.AddOrientationConstraint(...)
        # initial_guess = np.zeros(self.plant.num_positions())
        # q = ik.q()
        # prog = ik.get_mutable_prog()
        # prog.SetInitialGuess(q, initial_guess)
        # result = Solve(ik.prog())
        # if not result.is_success():
        #     print("IK failed")
        # q = result.GetSolution(q)

        # Weight the different parts of the cost function
        final_cost = 100*distance_obj_pc_centroid_to_X_OG_y_axis + angle

        return final_cost
    

    def SelectGrasp(self, context, output):

        obj_pc = self.get_input_port(0).Eval(context)
        obj_traj = self.get_input_port(1).Eval(context)

        self.meshcat.SetObject("cloud", obj_pc)

        obj_pc_centroid = np.mean(obj_pc.xyzs(), axis=1)  # column-wise mean of 3xN np array of points

        # Find range of time where object is likely within iiwa's work evelope
        self.obj_reachable_start_t = 0.5  # random guess
        self.obj_reachable_end_t = 1.0  # random guess
        search_times = np.linspace(0.5, 1, 20)  # assuming first half of trajectory is definitely outside of iiwa's work envelope
        # Forward search to find the first time that the object is in iiwa's work envelope
        for t in search_times:
            obj_pos = obj_traj.value(t)[:3]  # (3,1) np array containing x,y,z
            obj_dist_from_iiwa_squared = obj_pos[0][0]**2 + obj_pos[1][0]**2
            # Object is between 420-750mm from iiwa's center in XY plane
            if obj_dist_from_iiwa_squared > 0.42**2 and obj_dist_from_iiwa_squared < 0.75**2:
                self.obj_reachable_start_t = t
        # Backward search to find last time
        for t in search_times[::-1]:
            obj_pos = obj_traj.value(t)[:3]  # (3,1) np array containing x,y,z
            obj_dist_from_iiwa_squared = obj_pos[0][0]**2 + obj_pos[1][0]**2
            # Object is between 420-750mm from iiwa's center in XY plane
            if obj_dist_from_iiwa_squared > 0.42**2 and obj_dist_from_iiwa_squared < 0.75**2:
                self.obj_reachable_end_t = t
        
        # For now, all grasps will happen in the middle of when obj is in iiwa's work envelope
        obj_catch_t = 0.5*(self.obj_reachable_start_t + self.obj_reachable_end_t)

        start = time.time()
        grasp_candidates = self.compute_candidate_grasps(
            obj_pc, candidate_num=20, random_seed=12
        )
        print(time.time() - start)
        print(f"grasp_candidates: {grasp_candidates}")

        self.meshcat.Delete()
        self.meshcat.SetObject("cloud", obj_pc)

        """
        Iterate through all grasps and select the best based on the following heuristic:
        Minimum distance between the centroid of the object and the y-axis ray of X_WG.

        This ensures the gripper doesn't try to grab an edge of the object.
        """
        min_cost = float('inf')
        min_cost_grasp = None
        for i in range(len(grasp_candidates)):
            
            grasp_cost = grasp_cost(obj_pc_centroid, grasp_candidates[i], obj_catch_t)

            if grasp_cost < min_cost:
                min_cost = grasp_cost
                min_cost_grasp = {grasp_candidates[i], obj_catch_t}

            # draw all grasp candidates
            # draw_grasp_candidate(
            #     grasp_candidates[i], prefix="gripper" + str(i), draw_frames=False
            # )

        self.draw_grasp_candidate(
            min_cost_grasp, prefix="gripper_best", draw_frames=False
        )

        output.set_value(min_cost_grasp)