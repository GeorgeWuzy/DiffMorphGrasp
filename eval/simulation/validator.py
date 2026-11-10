from isaacgym import gymapi, gymtorch

import math
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch

from .controller import controller
from .rotation import relative_pose, to_euler


class IsaacValidator:
    def __init__(self, robot, robot_urdf, batch_size, gpu=0, use_gui=False):
        self.robot = robot
        self.robot_urdf = Path(robot_urdf).resolve()
        self.batch_size = batch_size
        self.chain = pk.build_chain_from_urdf(self.robot_urdf.read_bytes()).to(dtype=torch.float32, device="cpu")
        self.names = list(self.chain.get_joint_parameter_names())
        if len(self.names) < 6 or not all(n.startswith("virtual_joint") for n in self.names[:6]):
            raise ValueError("The stability test requires an extended hand URDF with six virtual root joints")
        self.gym = gymapi.acquire_gym()
        params = gymapi.SimParams()
        params.dt = 0.01
        params.substeps = 2
        params.gravity = gymapi.Vec3(0, 0, 0)
        params.physx.use_gpu = True
        params.physx.solver_type = 1
        params.physx.num_position_iterations = 8
        params.physx.num_velocity_iterations = 0
        params.physx.contact_offset = 0.01
        params.physx.rest_offset = 0.
        params.physx.max_depenetration_velocity = 1.
        self.sim = self.gym.create_sim(gpu, gpu if use_gui else -1, gymapi.SIM_PHYSX, params)
        if self.sim is None:
            raise RuntimeError("Isaac Gym could not create the GPU PhysX simulation")
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties()) if use_gui else None
        self.envs = []
        self.robot_handles = []
        self.object_handles = []
        self.diagnostics = {}

    def set_asset(self, object_urdf):
        object_urdf = Path(object_urdf).resolve()
        options = gymapi.AssetOptions()
        options.disable_gravity = True
        options.fix_base_link = True
        options.collapse_fixed_joints = True
        self.robot_asset = self.gym.load_asset(self.sim, str(self.robot_urdf.parent), self.robot_urdf.name, options)
        options = gymapi.AssetOptions()
        options.fix_base_link = False
        options.override_com = True
        options.override_inertia = True
        options.density = 500.
        self.object_asset = self.gym.load_asset(self.sim, str(object_urdf.parent), object_urdf.name, options)
        if self.robot_asset is None or self.object_asset is None:
            raise RuntimeError("Failed to load evaluation assets")
        if self.gym.get_asset_rigid_body_count(self.object_asset) != 1:
            raise ValueError("Evaluation expects a single rigid-body object")
        self.body_count = self.gym.get_asset_rigid_body_count(self.robot_asset) + 1

    def create_envs(self):
        indices = []
        for i in range(self.batch_size):
            env = self.gym.create_env(self.sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), max(1, math.isqrt(self.batch_size)))
            self.envs.append(env)
            obj = self.gym.create_actor(env, self.object_asset, gymapi.Transform(), f"object_{i}", i)
            hand = self.gym.create_actor(env, self.robot_asset, gymapi.Transform(), f"robot_{i}", i)
            self.object_handles.append(obj)
            self.robot_handles.append(hand)
            indices.append(self.gym.get_actor_rigid_body_index(env, obj, 0, gymapi.DOMAIN_SIM))
            for actor in (obj, hand):
                shapes = self.gym.get_actor_rigid_shape_properties(env, actor)
                for shape in shapes:
                    shape.friction = 3.
                self.gym.set_actor_rigid_shape_properties(env, actor, shapes)
            dof = self.gym.get_actor_dof_properties(env, hand)
            asset_names = list(self.gym.get_asset_dof_names(self.robot_asset))
            if set(asset_names) != set(self.names) or len(asset_names) != len(self.names):
                raise ValueError(f"Isaac/FK joint mismatch: {asset_names} vs {self.names}")
            root_mask = np.array([name.startswith("virtual_joint") for name in asset_names])
            dof["driveMode"].fill(gymapi.DOF_MODE_POS)
            dof["stiffness"] = np.where(root_mask, 1e6, 1000.)
            dof["damping"] = np.where(root_mask, 1e5, 50.)
            dof["armature"].fill(1e-3)
            self.gym.set_actor_dof_properties(env, hand, dof)
        self.to_isaac = [self.names.index(name) for name in asset_names]
        self.to_urdf = [asset_names.index(name) for name in self.names]
        self.object_indices = torch.tensor(indices, dtype=torch.long)
        mass = self.gym.get_actor_rigid_body_properties(self.envs[0], self.object_handles[0])[0].mass
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError(f"Invalid dynamic object mass: {mass}")
        self.force = 0.5 * mass
        self.diagnostics["object_mass_kg"] = mass
        self.diagnostics["force_n"] = self.force

    def set_actor_pose_dof(self, q):
        q = to_euler(q.cpu(), len(self.names))
        lower, upper = [torch.as_tensor(v) for v in self.chain.get_joint_limits()]
        violations = ((q[:, 6:] < lower[6:] - 1e-5) | (q[:, 6:] > upper[6:] + 1e-5)).any(dim=1)
        self.diagnostics['input_joint_limit_violation'] = violations.tolist()
        self.gym.prepare_sim(self.sim)
        self.body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        root = self.gym.acquire_actor_root_state_tensor(self.sim)
        states = gymtorch.wrap_tensor(root)
        states[:] = states.new_tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        self.gym.set_actor_root_state_tensor(self.sim, root)
        outer, inner = controller(self.robot, q, self.chain)
        for i, (env, hand) in enumerate(zip(self.envs, self.robot_handles)):
            state = self.gym.get_actor_dof_states(env, hand, gymapi.STATE_ALL).copy()
            state["pos"] = outer[i, self.to_isaac].numpy()
            state["vel"].fill(0)
            self.gym.set_actor_dof_states(env, hand, state, gymapi.STATE_ALL)
            self.gym.set_actor_dof_position_targets(env, hand, inner[i, self.to_isaac].numpy())

    def _step(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if self.viewer is not None:
            if self.gym.query_viewer_has_closed(self.viewer):
                raise RuntimeError("Evaluation interrupted before all forces were applied")
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

    def run_sim(self):
        for _ in range(100):
            self._step()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        object_pose = self.body_state[self.object_indices, :7].clone()
        start = object_pose[:, :3].clone()
        joints = self.dof_state.reshape(self.batch_size, -1, 2)[:, self.to_urdf, 0].clone()
        q_grasped = relative_pose(object_pose, joints)
        finite = torch.isfinite(q_grasped).all(dim=1) & torch.isfinite(object_pose).all(dim=1)
        forces = torch.zeros((self.batch_size * self.body_count, 3), dtype=torch.float32)
        maximum = torch.zeros(self.batch_size)
        boundary = []
        for axis, sign in ((0, 1), (1, 1), (2, 1), (0, -1), (1, -1), (2, -1)):
            forces.zero_()
            forces[self.object_indices, axis] = self.force * sign
            for _ in range(100):
                applied = self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), None, gymapi.ENV_SPACE)
                if applied is False:
                    raise RuntimeError("Isaac Gym rejected the external force tensor")
                self._step()
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                current_state = self.body_state[self.object_indices]
                current = current_state[:, :7]
                finite &= torch.isfinite(current_state).all(dim=1) & (current[:, 3:7].norm(dim=1) > 1e-12)
                distance = (current[:, :3] - start).norm(dim=-1)
                maximum = torch.maximum(maximum, distance)
            boundary.append(distance.tolist())
        success = finite & (distance <= 0.02)
        self.diagnostics.update({
            "final_displacement_m": distance.tolist(),
            "max_displacement_m": maximum.tolist(),
            "direction_boundary_displacement_m": boundary,
            "finite": finite.tolist(),
            "object_position_after_closure_m": start.tolist(),
        })
        return success, q_grasped

    def destroy(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
