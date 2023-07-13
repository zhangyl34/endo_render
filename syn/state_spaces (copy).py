import os
import time

import gym
import numpy as np
import scipy.stats as sstats
import trimesh
from autolab_core import Logger, RigidTransform
import math

from .random_variables import CameraRandomVariable
from .states import CameraState, HeapAndCameraState, HeapState, ObjectState

KEY_SEP_TOKEN = "~"

class CameraStateSpace(gym.Space):
    """State space for a camera."""

    def __init__(self, config):
        self._config = config

        # read params
        self.frame = config["name"]

        # random variable for pose of camera
        self.camera_rv = CameraRandomVariable(config)

    def sample(self):
        """Sample a camera state."""
        pose, intrinsics = self.camera_rv.sample(size=1)
        return CameraState(self.frame, pose, intrinsics)


class HeapStateSpace(gym.Space):
    """State space for object heaps."""

    def __init__(self, physics_engine, config):

        self._physics_engine = physics_engine
        self._config = config  # heap

        # set up logger
        # dataset_generation.log
        self._logger = Logger.get_logger(self.__class__.__name__)

        # Set up object configs
        self.obj_density = 4000

        # Setup object keys and directories
        object_keys = []
        object_keys.append('shuangji')
        object_keys.append('zuzhi')
        object_keys.append('chizhen')
        self.all_object_keys = list(np.array(object_keys))

    @property
    def obj_keys(self):
        return self.all_object_keys

    def sample(self, tools: dict):
        """Samples a state from the space
        Returns
        -------
        :obj:`HeapState`
            state of the object pile
        """

        # Start physics engine
        self._physics_engine.start()

        """ setup workspace."""
        workspace_obj_states = []
        if "shuangji" in tools:
            # load mesh
            # read subconfigs
            mesh_dirname = self._config['objects']['mesh_dir'] + '/shuangjizhuaqian'
            # 更改为绝对路径
            if not os.path.isabs(mesh_dirname):
                mesh_dirname = os.path.join(os.getcwd(), mesh_dirname)
            
            # 1
            mesh_filename = os.path.join(mesh_dirname, 'xiaohaojizuo.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T0 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([tools['shuangji']['trans_x'], tools['shuangji']['trans_y'], tools['shuangji']['trans_z']]),
                from_frame="world",
                to_frame="shuangji_0",
            )
            T1 = RigidTransform(
                rotation=RigidTransform.z_axis_rotation(tools['shuangji']['rot_z']*np.pi/180),
                translation=np.array([0.0,0.0,0.0]),
                from_frame="shuangji_0",
                to_frame="shuangji_1",
            )
            T2 = RigidTransform(
                rotation=RigidTransform.y_axis_rotation(tools['shuangji']['rot_y']*np.pi/180),
                translation=np.array([0.0,0.0,0.0]),
                from_frame="shuangji_1",
                to_frame="shuangji_2",
            )
            T3 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation((tools['shuangji']['rot_x'])*np.pi/180),
                translation=np.array([0.0,0.0,0.0]),
                from_frame="shuangji_2",
                to_frame="shuangji_3",
            )
            T4 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(np.pi/2),
                translation=np.array([0.0,0.0,0.0]),
                from_frame="shuangji_3",
                to_frame="shuangji1_4",
            )
            pose = T4*T3*T2*T1*T0
            workspace_obj = ObjectState('shuangji1', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

            # 2
            mesh_filename = os.path.join(mesh_dirname, 'dingqiantou.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([0.0,0.0,6.43]),
                from_frame="shuangji_3",
                to_frame="shuangji2_4",
            )            
            T5 = RigidTransform(
                rotation=RigidTransform.z_axis_rotation(np.pi/2),
                translation=np.array([0.0,0.0,0.0]),
                from_frame="shuangji2_4",
                to_frame="shuangji2_5",
            )            
            T6 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(np.pi),
                translation=np.array([0.0,0.0,0.0]),
                from_frame="shuangji2_5",
                to_frame="shuangji2_6",
            )
            pose = T6*T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('shuangji2', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

            # 3
            mesh_filename = os.path.join(mesh_dirname, 'jueyuanjian.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([-1.6, 0, 9.9]),
                from_frame="shuangji_3",
                to_frame="shuangji3_4",
            )            
            T5 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(np.pi/2),
                translation=np.array([0.0,0.0,0.0]),
                from_frame="shuangji3_4",
                to_frame="shuangji3_5",
            )            
            T6 = RigidTransform(
                rotation=RigidTransform.y_axis_rotation(-np.pi),
                translation=np.array([0.0,0.0,0.0]),
                from_frame="shuangji3_5",
                to_frame="shuangji3_6",
            )
            pose = T6*T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('shuangji3', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

            # 4
            mesh_filename = os.path.join(mesh_dirname, 'dongqiantou.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([1.9, 0, 5.7]),
                from_frame="shuangji_3",
                to_frame="shuangji4_4",
            )            
            T5 = RigidTransform(
                rotation=RigidTransform.y_axis_rotation(tools['shuangji']['angle']),
                translation=np.array([0.0, 0.0, 0.0]),
                from_frame="shuangji4_4",
                to_frame="shuangji4_5",
            )            
            T6 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([-0.5, 0, 13]),
                from_frame="shuangji4_5",
                to_frame="shuangji4_6",
            )
            T7 = RigidTransform(
                rotation=RigidTransform.z_axis_rotation(-np.pi/2),
                translation=np.array([0,0,0]),
                from_frame="shuangji4_6",
                to_frame="shuangji4_7",
            )
            T8 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(np.pi/2),
                translation=np.array([0,0,0]),
                from_frame="shuangji4_7",
                to_frame="shuangji4_8",
            )
            pose = T8*T7*T6*T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('shuangji4', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

            # 5
            mesh_filename = os.path.join(mesh_dirname, 'gudingxiao.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([1.9, 0, 5.7]),
                from_frame="shuangji_3",
                to_frame="shuangji5_4",
            )
            T5 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(np.pi/2),
                translation=np.array([0,0,0]),
                from_frame="shuangji5_4",
                to_frame="shuangji5_5",
            )
            pose = T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('shuangji5', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

            # 6
            mesh_filename = os.path.join(mesh_dirname, 'gudingxiao.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([0, 0, 5.7 - 1.9/math.tan((tools['shuangji']['angle']/180*math.pi)+math.atan(1.9/3.75))]),
                from_frame="shuangji_3",
                to_frame="shuangji6_4",
            )
            T5 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(np.pi/2),
                translation=np.array([0,0,0]),
                from_frame="shuangji6_4",
                to_frame="shuangji6_5",
            )
            pose = T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('shuangji6', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

            # 7
            mesh_filename = os.path.join(mesh_dirname, 'qudonghuakuai.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([-0.05, -2.54, -3.34 - 1.9/math.tan((tools['shuangji']['angle']/180*math.pi)+math.atan(1.9/3.75)) + 3.75]),
                from_frame="shuangji_3",
                to_frame="shuangji7_4",
            )
            T5 = RigidTransform(
                rotation=RigidTransform.z_axis_rotation(-np.pi/2),
                translation=np.array([0,0,0]),
                from_frame="shuangji7_4",
                to_frame="shuangji7_5",
            )
            pose = T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('shuangji7', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

            # 8
            mesh_filename = os.path.join(mesh_dirname, 'xianguan.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([-3.05, 0, 7.89]),
                from_frame="shuangji_3",
                to_frame="shuangji8_4",
            )
            T5 = RigidTransform(
                rotation=RigidTransform.y_axis_rotation((90-4.85391)*np.pi/180),
                translation=np.array([0,0,0]),
                from_frame="shuangji8_4",
                to_frame="shuangji8_5",
            )
            pose = T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('shuangji8', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

            # 9
            mesh_filename = os.path.join(mesh_dirname, '../jueyuanzuo.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.z_axis_rotation((-20.93)*np.pi/180),
                translation=np.array([0,0,0]),
                from_frame="shuangji_3",
                to_frame="shuangji9_4",
            )            
            T5 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([-7.69796, 0, -5.63]),
                from_frame="shuangji9_4",
                to_frame="shuangji9_5",
            )
            T6 = RigidTransform(
                rotation=RigidTransform.y_axis_rotation((90)*np.pi/180),
                translation=np.array([0,0,0]),
                from_frame="shuangji9_5",
                to_frame="shuangji9_6",
            )
            pose = T6*T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('shuangji9', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

            # 10
            mesh_filename = os.path.join(mesh_dirname, '../lianxutibi/arm.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([0, 0, -26]),
                from_frame="shuangji_3",
                to_frame="shuangji10_4",
            )
            T5 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation((90)*np.pi/180),
                translation=np.array([0,0,0]),
                from_frame="shuangji10_4",
                to_frame="shuangji10_5",
            )
            pose = T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('shuangji10', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

            # 11
            mesh_filename = os.path.join(mesh_dirname, '../lianxutibi/arm.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([0, 0, -75]),
                from_frame="shuangji_3",
                to_frame="shuangji11_4",
            )
            T5 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation((90)*np.pi/180),
                translation=np.array([0,0,0]),
                from_frame="shuangji11_4",
                to_frame="shuangji11_5",
            )
            pose = T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('shuangji11', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

            # 12
            mesh_filename = os.path.join(mesh_dirname, '../lianxutibi/arm.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([0, 0, -100]),
                from_frame="shuangji_3",
                to_frame="shuangji12_4",
            )
            T5 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation((90)*np.pi/180),
                translation=np.array([0,0,0]),
                from_frame="shuangji12_4",
                to_frame="shuangji12_5",
            )
            pose = T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('shuangji12', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

        if "zuzhi" in tools:
            # load mesh
            # read subconfigs
            mesh_dirname = self._config['objects']['mesh_dir'] + '/zuzhizhuaqian'
            # 更改为绝对路径
            if not os.path.isabs(mesh_dirname):
                mesh_dirname = os.path.join(os.getcwd(), mesh_dirname)
            
            # 1
            mesh_filename = os.path.join(mesh_dirname, 'zuzhi_ding.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T0 = RigidTransform(
                rotation=RigidTransform.z_axis_rotation(0.0),
                translation=np.array([tools['zuzhi']['trans_x'], tools['zuzhi']['trans_y'], tools['zuzhi']['trans_z']]),
                from_frame="world",
                to_frame="zuzhi_0",
            )
            T1 = RigidTransform(
                rotation=RigidTransform.z_axis_rotation(tools['zuzhi']['rot_z']*np.pi/180),
                translation=np.array([0.0,0.0,0.0]),
                from_frame="zuzhi_0",
                to_frame="zuzhi_1",
            )
            T2 = RigidTransform(
                rotation=RigidTransform.y_axis_rotation(tools['zuzhi']['rot_y']*np.pi/180),
                translation=np.array([0.0,0.0,0.0]),
                from_frame="zuzhi_1",
                to_frame="zuzhi_2",
            )
            T3 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation((tools['zuzhi']['rot_x'])*np.pi/180),
                translation=np.array([0.0,0.0,0.0]),
                from_frame="zuzhi_2",
                to_frame="zuzhi_3",
            )
            T4 = RigidTransform(
                rotation=RigidTransform.z_axis_rotation(0.0),
                translation=np.array([-3.25, -3.25, -3]),
                from_frame="zuzhi_3",
                to_frame="zuzhi1_4",
            )
            T5 = RigidTransform(
                rotation=RigidTransform.y_axis_rotation(-np.pi/2),
                translation=np.array([0.0,0.0,0.0]),
                from_frame="zuzhi1_4",
                to_frame="zuzhi1_5",
            )
            pose = T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('zuzhi1', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

            # 2
            mesh_filename = os.path.join(mesh_dirname, 'zuzhi_dong_1.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.z_axis_rotation(0.0),
                translation=np.array([0, 0, 8.7]),
                from_frame="zuzhi_3",
                to_frame="zuzhi2_4",
            )
            T5 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(tools['zuzhi']['angle']/2*np.pi/180),
                translation=np.array([0.0,0.0,0.0]),
                from_frame="zuzhi2_4",
                to_frame="zuzhi2_5",
            )            
            T6 = RigidTransform(
                rotation=RigidTransform.z_axis_rotation(0.0),
                translation=np.array([-2.7, -1.5, -11.7]),
                from_frame="zuzhi2_5",
                to_frame="zuzhi2_6",
            )
            T7 = RigidTransform(
                rotation=RigidTransform.y_axis_rotation(-np.pi/2),
                translation=np.array([0.0,0.0,0.0]),
                from_frame="zuzhi2_6",
                to_frame="zuzhi2_7",
            )
            pose = T7*T6*T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('zuzhi2', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

            # 3
            mesh_filename = os.path.join(mesh_dirname, '../jueyuanzuo.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.z_axis_rotation((-20.93)*np.pi/180),
                translation=np.array([0, 0, 0]),
                from_frame="zuzhi_3",
                to_frame="zuzhi3_4",
            )
            T5 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([-7.69796, 0, -5.63]),
                from_frame="zuzhi3_4",
                to_frame="zuzhi3_5",
            )            
            T6 = RigidTransform(
                rotation=RigidTransform.y_axis_rotation(np.pi/2),
                translation=np.array([0,0,0]),
                from_frame="zuzhi3_5",
                to_frame="zuzhi3_6",
            )
            pose = T6*T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('zuzhi3', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)


            # 4
            mesh_filename = os.path.join(mesh_dirname, '../lianxutibi/arm.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([0, 0, -26]),
                from_frame="zuzhi_3",
                to_frame="zuzhi4_4",
            )
            T5 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation((90)*np.pi/180),
                translation=np.array([0,0,0]),
                from_frame="zuzhi4_4",
                to_frame="zuzhi4_5",
            )
            pose = T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('zuzhi4', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

            # 5
            mesh_filename = os.path.join(mesh_dirname, '../lianxutibi/arm.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([0, 0, -75]),
                from_frame="zuzhi_3",
                to_frame="zuzhi5_4",
            )
            T5 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation((90)*np.pi/180),
                translation=np.array([0,0,0]),
                from_frame="zuzhi5_4",
                to_frame="zuzhi5_5",
            )
            pose = T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('zuzhi5', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

            # 6
            mesh_filename = os.path.join(mesh_dirname, '../lianxutibi/arm.obj')
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            T4 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation(0),
                translation=np.array([0, 0, -100]),
                from_frame="zuzhi_3",
                to_frame="zuzhi6_4",
            )
            T5 = RigidTransform(
                rotation=RigidTransform.x_axis_rotation((90)*np.pi/180),
                translation=np.array([0,0,0]),
                from_frame="zuzhi6_4",
                to_frame="zuzhi6_5",
            )
            pose = T5*T4*T3*T2*T1*T0
            workspace_obj = ObjectState('zuzhi6', mesh, pose)
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

        # Stop physics engine
        self._physics_engine.stop()

        return HeapState(workspace_obj_states)


class HeapAndCameraStateSpace(gym.Space):
    """State space for environments."""

    def __init__(self, physics_engine, config):

        heap_config = config["heap"]
        cam_config = config["camera"]

        # individual state spaces
        self.heap = HeapStateSpace(physics_engine, heap_config)
        self.camera = CameraStateSpace(cam_config)

    @property
    def obj_keys(self):
        return self.heap.obj_keys

    def sample(self, tools: dict):
        """Sample a state."""
        # sample individual states
        heap_state = self.heap.sample(tools)
        cam_state = self.camera.sample()  # 假 sample

        return HeapAndCameraState(heap_state, cam_state)
