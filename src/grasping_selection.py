from pydrake.all import (
    AbstractValue,
    Concatenate,
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
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddMultibodyTriad
from manipulation.utils import ConfigureParser

import time
import numpy as np
from scipy.spatial import KDTree

from utils import ObjectTrajectory


class GraspSelector(LeafSystem):
    """
    Use method described in this paper: https://arxiv.org/pdf/1706.09911.pdf to
    sample potential grasps until finding one at a desirable position for iiwa.
    """

    def __init__(self, plant, scene_graph, meshcat, thrown_model_name, grasp_random_seed):
        LeafSystem.__init__(self)

        obj_pc = AbstractValue.Make(PointCloud())
        obj_traj = AbstractValue.Make(ObjectTrajectory())
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
        self.thrown_model_name = thrown_model_name
        self.grasp_random_seed = grasp_random_seed
        self.random_transform = RigidTransform([-1, -1, 1])  # used for visualizing grasp candidates off to the side
        self.selected_grasp_obj_frame = None
        self.obj_catch_t = None
        self.visualize = True


    def draw_grasp_candidate(self, X_G, prefix="gripper", random_transform=True):
        """
        Helper function to visualize grasp.
        """
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        parser = Parser(plant)
        ConfigureParser(parser)
        gripper = parser.AddModelsFromUrl(
            "package://manipulation/schunk_wsg_50_welded_fingers.sdf"
        )

        if random_transform:
            X_G = self.random_transform @ X_G

        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body"), X_G)
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
        TODO: Speed up this function. Setting up the diagram takes ~0.1 sec,
        actually computing SDF is also ~0.1 sec

        Builds a new MBP and diagram with just the object and WSG, and computes
        SDF to check if there is collision.
        """
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        parser = Parser(plant)
        ConfigureParser(parser)
        parser.AddModelsFromUrl(
            "package://manipulation/schunk_wsg_50_welded_fingers.sdf"
        )
        AddMultibodyTriad(plant.GetFrameByName("body"), scene_graph)
        plant.Finalize()

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        plant_context = plant.GetMyContextFromRoot(context)
        scene_graph_context = scene_graph.GetMyContextFromRoot(context)
        plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_G)

        query_object = scene_graph.get_query_output_port().Eval(
            scene_graph_context
        )

        for pt in obj_pc.xyzs().T:
            distances = query_object.ComputeSignedDistanceToPoint(pt)
            for body_index in range(len(distances)):
                distance = distances[body_index].distance
                if distance < 0:
                    return True  # Collision

        return False


    def compute_darboux_frame(self, index, obj_pc, kdtree, ball_radius=0.002, max_nn=50):
        """
        Given a index of the pointcloud, return a RigidTransform from origin of
        point cloud to the Darboux frame at that point.

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

        # 6. Check if matrix is improper (is actually both a rotation and reflection), if so, fix it
        if np.linalg.det(R) < 0:  # if det is neg, this means rot matrix is improper
            R[:, 0] *= -1  # multiply left column (v2) by -1 to fix improperness

        #7. Create a rigid transform with the rotation of the normal and position of the point in the PC
        X_OF = RigidTransform(RotationMatrix(R), points[:,index])  # modify here.

        return X_OF



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


    def compute_grasp_cost(self, obj_pc_centroid, X_OG, t):
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
        X_WO = self.obj_pose_at_catch
        X_WG = X_WO @ X_OG

        # Add cost associated with whether X_WG's y-axis points away from iiwa (which is what we want)
        world_z_axis_to_X_WG_vector = np.append(X_WG.translation()[:2], 0)  # basically replacing z with 0
        world_z_axis_to_X_WG_vector = world_z_axis_to_X_WG_vector / np.linalg.norm(world_z_axis_to_X_WG_vector)
        X_WG_y_axis_vector = (X_WG.rotation().matrix() @ np.array([[0],[1],[0]])).reshape((3,))
        # # On the order of 0 - 2
        direction = 1 - np.dot(world_z_axis_to_X_WG_vector, X_WG_y_axis_vector)

        # Add cost associated with whether object is able to fly in between two fingers of gripper
        # Z-axis of gripper should be aligned with derivative of obj trajectory
        obj_vel_at_catch = self.obj_traj.EvalDerivative(t)[:3]  # (3,) np array
        obj_direction_at_catch = obj_vel_at_catch / np.linalg.norm(obj_vel_at_catch)  # normalize
        X_WG_z_axis_vector = (X_WG.rotation().matrix() @ np.array([[0],[0],[1]])).reshape((3,))
        # on the order of 0 - 2
        alignment = 1 - np.absolute(np.dot(obj_direction_at_catch, X_WG_z_axis_vector))  # absolute since it's ok for gripper z-axis to be perfectly against obj velocity

        # if (alignment < 0.010 and direction < 0.040):
        #     print(f"\nworld_z_axis_to_X_WG_vector: {world_z_axis_to_X_WG_vector}")
        #     print(f"X_WG_y_axis_vector: {X_WG_y_axis_vector}")
        #     print(f"direction: {direction}")
        #     print(f"\nobj_direction_at_catch: {obj_direction_at_catch}")
        #     print(f"X_WG_z_axis_vector: {X_WG_z_axis_vector}")
        #     print(f"alignment: {alignment}\n")

        # Weight the different parts of the cost function
        final_cost = alignment + direction + 10*distance_obj_pc_centroid_to_X_OG_y_axis

        return final_cost, distance_obj_pc_centroid_to_X_OG_y_axis, direction, alignment


    def compute_candidate_grasps(self, obj_pc, obj_pc_centroid, obj_catch_t, candidate_num=2000):
        """
        Args:
            - obj_pc (PointCloud object): pointcloud of the object.
            - candidate_num (int) : number of desired candidates.
        Return:
            - candidate_lst (list of drake RigidTransforms) : candidate list of grasps.
        """

        # Constants for random variation
        np.random.seed(self.grasp_random_seed)

        # Build KD tree for the pointcloud.
        kdtree = KDTree(obj_pc.xyzs().T)
        ball_radius = 0.002

        candidate_lst = {}  # dict mapping candidates (given by RigidTransforms) to cost of that candidate

        def compute_candidate(idx, obj_pc, kdtree, ball_radius, candidate_lst_lock, candidate_lst):
            X_OF = self.compute_darboux_frame(idx, obj_pc, kdtree, ball_radius)  # find Darboux frame of random point

            # offset gripper pose from object centroid depending on object size
            if "banana" in self.thrown_model_name.lower():
                y_offset = -0.04
            if "ball" in self.thrown_model_name.lower():
                y_offset = -0.05
            if "bottle" in self.thrown_model_name.lower():
                y_offset = -0.05
            new_X_OG = X_OF @ RigidTransform(np.array([0, y_offset, 0]))  # Move gripper back by fixed amount

            grasp_CoM_cost_threshold = 0.030  # range: 0 - 0.05
            direction_cost_threshold = 0.400  # range: 0 - 2
            collision_cost_threshold = 0.100  # range: 0 - 2
            new_X_OG_cost, grasp_CoM_cost, direction_cost, collision_cost = self.compute_grasp_cost(obj_pc_centroid, new_X_OG, obj_catch_t)
            # if grasp isn't above thresholds, don't even bother checking for collision (which is slow)
            if grasp_CoM_cost > grasp_CoM_cost_threshold or direction_cost > direction_cost_threshold or collision_cost > collision_cost_threshold:
                return

            # print("passed grasping thresholds")

            # check_collision takes most of the runtime
            if (self.check_collision(obj_pc, new_X_OG) is not True) and self.check_nonempty(obj_pc, new_X_OG):  # no collision, and there is an object between fingers
                with candidate_lst_lock:
                    candidate_lst[new_X_OG] = new_X_OG_cost

        import threading

        threads = []
        candidate_lst_lock = threading.Lock()
        for _ in range(candidate_num):
            random_idx = np.random.randint(0, obj_pc.size())
            t = threading.Thread(target=compute_candidate, args=(random_idx,
                                                                obj_pc,
                                                                kdtree,
                                                                ball_radius,
                                                                candidate_lst_lock,
                                                                candidate_lst))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if len(candidate_lst) == 0:
            print("grasp sampling did not find any valid candidates.")

        print(f"found {len(candidate_lst)} potential grasps")

        return candidate_lst


    def SelectGrasp(self, context, output):
        if self.selected_grasp_obj_frame is None:
            self.obj_pc = self.get_input_port(0).Eval(context).VoxelizedDownSample(voxel_size=0.0025)
            self.obj_pc.EstimateNormals(0.05, 30)  # allows us to use obj_pc.normals() function later
            self.obj_traj = self.get_input_port(1).Eval(context)

            if (self.obj_traj == ObjectTrajectory()):  # default output of TrajectoryPredictor system; means that it hasn't seen the object yet
                print("received default traj (in SelectGrasp)")
                return

            self.meshcat.SetObject("cloud", self.obj_pc)

            obj_pc_centroid = np.mean(self.obj_pc.xyzs(), axis=1)  # column-wise mean of 3xN np array of points

            # Find range of time where object is likely within iiwa's work evelope
            self.obj_reachable_start_t = 0.5  # random guess
            self.obj_reachable_end_t = 1.0  # random guess
            search_times = np.linspace(0.5, 1, 20)  # assuming first half of trajectory is definitely outside of iiwa's work envelope
            # Forward search to find the first time that the object is in iiwa's work envelope
            for t in search_times:
                obj_pose = self.obj_traj.value(t)
                obj_pos = obj_pose.translation()  # (3,) np array containing x,y,z
                obj_dist_from_iiwa_squared = obj_pos[0]**2 + obj_pos[1]**2
                # Object is between 420-750mm from iiwa's center in XY plane
                if obj_dist_from_iiwa_squared > 0.42**2 and obj_dist_from_iiwa_squared < 0.75**2:
                    self.obj_reachable_start_t = t
            # Backward search to find last time
            for t in search_times[::-1]:
                obj_pose = self.obj_traj.value(t)
                obj_pos = obj_pose.translation()  # (3,) np array containing x,y,z
                obj_dist_from_iiwa_squared = obj_pos[0]**2 + obj_pos[1]**2
                # Object is between 420-750mm from iiwa's center in XY plane
                if obj_dist_from_iiwa_squared > 0.42**2 and obj_dist_from_iiwa_squared < 0.75**2:
                    self.obj_reachable_end_t = t

            # For now, all grasps will happen at 0.475 of when obj is in iiwa's work envelope
            obj_catch_t = 0.475*(self.obj_reachable_start_t + self.obj_reachable_end_t)

            self.obj_pose_at_catch = self.obj_traj.value(obj_catch_t)

            start = time.time()
            grasp_candidates = self.compute_candidate_grasps(self.obj_pc, obj_pc_centroid, obj_catch_t)
            print(f"-----------grasp sampling runtime: {time.time() - start}")

            # Visualize point cloud
            obj_pc_for_visualization = PointCloud(self.obj_pc)
            if (self.visualize):
                obj_pc_for_visualization.mutable_xyzs()[:] = self.random_transform @ obj_pc_for_visualization.xyzs()
                self.meshcat.SetObject("cloud", obj_pc_for_visualization)

            """
            Iterate through all grasps and select the best based on the heuristics in compute_grasp_cost
            """
            min_cost = float('inf')
            min_cost_grasp = None  # RigidTransform, in object frame
            for grasp, grasp_cost in grasp_candidates.items():

                if grasp_cost < min_cost:
                    min_cost = grasp_cost
                    min_cost_grasp = grasp

                # draw all grasp candidates
                if (self.visualize):
                    self.draw_grasp_candidate(grasp, prefix="gripper " + str(time.time()))

            # Convert min_cost_grasp to world frame
            min_cost_grasp_W = self.obj_pose_at_catch @ min_cost_grasp

            # draw best grasp gripper position in world
            if (self.visualize):
                self.draw_grasp_candidate(min_cost_grasp_W, prefix="gripper_best", random_transform=False)

            output.set_value({min_cost_grasp_W: obj_catch_t})

            # Update class attributes so that next time grasp is not re-selected
            self.selected_grasp_obj_frame = min_cost_grasp
            self.obj_catch_t = obj_catch_t

        else:
            self.obj_traj = self.get_input_port(1).Eval(context)

            # Allow the estimated catch pose to vary in translation, not rotation, since rotation has so much noise that traj opt will fail
            estimated_obj_catch_pose = RigidTransform(self.obj_pose_at_catch.rotation(), self.obj_traj.value(self.obj_catch_t).translation())

            # Shift selected grasp slightly if object's predicted location at catch time has changed
            selected_grasp_world_frame = estimated_obj_catch_pose @ self.selected_grasp_obj_frame
            output.set_value({selected_grasp_world_frame: self.obj_catch_t})