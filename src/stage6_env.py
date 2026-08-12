"""Two colored balls on a robosuite table, registered as ColorBallPlace."""
from __future__ import annotations
import numpy as np
from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler

class ColorBallPlace(Stack):
    def _load_model(self):
        # Call ManipulationEnv implementation, bypassing Stack's cube construction.
        super(Stack, self)._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        tex = {"type": "cube"}
        mat = {"texrepeat": "1 1", "specular": "0.3", "shininess": "0.1"}
        red = CustomMaterial(
            texture="WoodRed", tex_name="ball_red_tex", mat_name="ball_red_mat",
            tex_attrib=tex, mat_attrib=mat,
        )
        green = CustomMaterial(
            texture="WoodGreen", tex_name="ball_green_tex", mat_name="ball_green_mat",
            tex_attrib=tex, mat_attrib=mat,
        )
        # Keep cubeA / cubeB attribute names so Stack's observables remain compatible.
        self.cubeA = BallObject(
            name="cubeA", size=[0.018], rgba=[1, 0, 0, 1],
            friction=[2.0, 0.02, 0.02], material=red,
        )
        self.cubeB = BallObject(
            name="cubeB", size=[0.018], rgba=[0, 1, 0, 1],
            friction=[2.0, 0.02, 0.02], material=green,
        )
        objects = [self.cubeA, self.cubeB]
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(objects)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="BallSampler",
                mujoco_objects=objects,
                x_range=[-0.10, 0.10],
                y_range=[-0.10, 0.10],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

    def _setup_references(self):
        """Add light drag so a released 24 g ball cannot roll indefinitely."""
        super()._setup_references()
        for ball in (self.cubeA, self.cubeB):
            joint_id = self.sim.model.joint_name2id(ball.joints[0])
            dof_address = int(self.sim.model.jnt_dofadr[joint_id])
            self.sim.model.dof_damping[dof_address:dof_address + 3] = 0.15
            self.sim.model.dof_damping[dof_address + 3:dof_address + 6] = 0.003
