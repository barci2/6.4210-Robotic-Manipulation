from pydrake.all import (
    AbstractValue,
    Concatenate,
    Trajectory,
    InputPortIndex,
    LeafSystem,
    PointCloud,
    RandomGenerator,
    RigidTransform,
    RollPitchYaw,
    Simulator,
    StartMeshcat,
    UniformlyRandomRotationMatrix,
)
import numpy as np


class GraspSelector(LeafSystem):
    """
    Use method described in this paper: https://arxiv.org/pdf/1706.09911.pdf to
    sample potential grasps until finding one at a desirable position for iiwa.
    """

    def __init__(self):
        """
        Args:
            xx
        """
        LeafSystem.__init__(self)
        obj_pc = AbstractValue.Make(PointCloud(0))
        obj_traj = AbstractValue.Make(Trajectory())
        self.DeclareAbstractInputPort("object_pc", obj_pc)
        self.DeclareAbstractInputPort("object_trajectory", obj_traj)
        
        port = self.DeclareAbstractOutputPort(
            "grasp_selection",
            lambda: AbstractValue.Make([RigidTransform()]),  # list of good candidate grasps
            self.SelectGrasp,
        )
        port.disable_caching_by_default()

        self._rng = np.random.default_rng()


    def SelectGrasp(self, context, output):
        body_poses = self.get_input_port(3).Eval(context)
        pcd = []
        for i in range(3):
            cloud = self.get_input_port(i).Eval(context)
            pcd.append(cloud.Crop(self._crop_lower, self._crop_upper))
            pcd[i].EstimateNormals(radius=0.1, num_closest=30)

            # Flip normals toward camera
            X_WC = body_poses[self._camera_body_indices[i]]
            pcd[i].FlipNormalsTowardPoint(X_WC.translation())
        merged_pcd = Concatenate(pcd)
        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)

        costs = []
        X_Gs = []
        # TODO(russt): Take the randomness from an input port, and re-enable
        # caching.
        for i in range(2):
            cost, X_G = GenerateAntipodalGraspCandidate(
                self._internal_model,
                self._internal_model_context,
                down_sampled_pcd,
                self._rng,
            )
            if np.isfinite(cost):
                costs.append(cost)
                X_Gs.append(X_G)

        if len(costs) == 0:
            # Didn't find a viable grasp candidate
            X_WG = RigidTransform(
                RollPitchYaw(-np.pi / 2, 0, np.pi / 2), [0.5, 0, 0.22]
            )
            output.set_value((np.inf, X_WG))
        else:
            best = np.argmin(costs)
            output.set_value((costs[best], X_Gs[best]))
