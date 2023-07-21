import gc  # 清除内存，尽量避免使用
import random
import os
import numpy as np

from autolab_core import (
    BinaryImage,
    ColorImage,
    DepthImage,
    GrayscaleImage,
    Logger,
    TensorDataset,
    YamlConfig,
)

from syn.bin_heap_env import BinHeapEnv
import syn.syn_utils as utils

from scipy.signal import convolve2d
from scipy import signal
from scipy.spatial.transform import Rotation

from plyfile import PlyData, PlyElement
from autolab_core import RigidTransform

# tools = {
#     'shuangji': {
#         'trans_x': 0,
#         'trans_y': 0,
#         'trans_z': 0,
#         'rot_x': 0,
#         'rot_y': 0,
#         'rot_z': 0,
#         'angle': 0,
#         'only_stem': False,
#         'theta_1': 0,
#         'rot_z_stem': 0
#     },
#     'zuzhi': {
#         'trans_x': 0,
#         'trans_y': 0,
#         'trans_z': 0,
#         'rot_x': 0,
#         'rot_y': 0,
#         'rot_z': 0,
#         'angle': 0,
#         'only_stem': False,
#         'theta_1': 0,
#         'rot_z_stem': 0
#     }
# }

tools = {
    'chizhen': {
        'trans_x': 0,
        'trans_y': 0,
        'trans_z': 0,
        'rot_x': 0,
        'rot_y': 0,
        'rot_z': 0,
        'angle': 0,
        'only_stem': False,
        'theta_1': 0,
        'rot_z_stem': 0
    },
}

def record_data(tools: dict, tool: str, data_file):
    data_file.write(str(tools[tool]['trans_x']) + '\t')
    data_file.write(str(tools[tool]['trans_y']) + '\t')
    data_file.write(str(tools[tool]['trans_z']) + '\t')
    data_file.write(str(tools[tool]['rot_x']) + '\t')
    data_file.write(str(tools[tool]['rot_y']) + '\t')
    data_file.write(str(tools[tool]['rot_z']) + '\t')
    data_file.write(str(tools[tool]['angle']) + '\n')


def depth2xyz(depth_map, mask, mask2, chizhen, f, cx, cy):
    ''' mask 为末端执行器整体 0 or 255
    mask2 为动钳头 0 or 255
    positive 区域为动钳头的主体部分
    '''

    # 末端执行器整体
    depth_map = np.where(mask>100, depth_map, 0.0)

    # 1. 随机向外膨胀 0~4 pixel
    kernel1 = np.asarray([[0,1,0],[1,1,1],[0,1,0]])
    # kernel2 = np.asarray([[0,0,1,0,0],[0,1,1,1,0],[0,0,1,0,0],[0,1,1,1,0],[0,0,1,0,0]])
    # kernel3 = np.asarray([[0,0,0,1,0,0,0],[0,0,1,1,1,0,0],[0,1,1,1,1,1,0],[1,1,1,1,1,1,1],[0,1,1,1,1,1,0],[0,0,1,1,1,0,0],[0,0,0,1,0,0,0]])
    # kernel4 = np.asarray([[0,0,0,0,1,0,0,0,0],[0,0,0,1,1,1,0,0,0],[0,0,1,1,1,1,1,0,0],[0,1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1,1],[0,1,1,1,1,1,1,1,0],[0,0,1,1,1,1,1,0,0],[0,0,0,1,1,1,0,0,0],[0,0,0,0,1,0,0,0,0]])
    mask_dilate1 = convolve2d(mask.astype(int), kernel1.astype(int), mode='same').astype(bool)
    mask_dilate1 = np.where(np.random.rand(mask.shape[0],mask.shape[1])>0.9, False, mask_dilate1)
    mask_dilate2 = convolve2d(mask_dilate1.astype(int), kernel1.astype(int), mode='same').astype(bool)
    mask_dilate2 = np.where(np.random.rand(mask.shape[0],mask.shape[1])>0.9, False, mask_dilate2)
    mask_dilate3 = convolve2d(mask_dilate2.astype(int), kernel1.astype(int), mode='same').astype(bool)
    mask_dilate3 = np.where(np.random.rand(mask.shape[0],mask.shape[1])>0.8, False, mask_dilate3)
    mask_dilate4 = convolve2d(mask_dilate3.astype(int), kernel1.astype(int), mode='same').astype(bool)
    mask_dilate4 = np.where(np.random.rand(mask.shape[0],mask.shape[1])>0.6, False, mask_dilate4)
    mask_dilate4 = np.where((depth_map>0.01), False, mask_dilate4)  # 只保留膨胀出来的区域为 True

    max_filter1 = signal.order_filter(depth_map+0.05, kernel1, 4)    # 邻域最大值
    max_filter2 = signal.order_filter(max_filter1+0.05, kernel1, 4)  # 邻域最大值
    max_filter3 = signal.order_filter(max_filter2+0.05, kernel1, 4)  # 邻域最大值
    max_filter4 = signal.order_filter(max_filter3+0.05, kernel1, 4)  # 邻域最大值
    depth_dilate4 = np.where(mask_dilate4, max_filter4, 0.0)
    depth_map = depth_map + depth_dilate4

    # 2. 再随机向外膨胀 1 pixel 作为 edge effect
    mask_dilate5 = convolve2d(mask_dilate4.astype(int), kernel1.astype(int), mode='same').astype(bool)
    mask_dilate5 = np.where(np.random.rand(mask.shape[0],mask.shape[1])>0.5, False, mask_dilate5)
    mask_dilate5 = np.where((depth_map>0.01), False, mask_dilate5)  # 只保留膨胀出来的区域为 True

    max_filter5 = signal.order_filter(depth_map, kernel1, 4)  # 邻域最大值
    depth_dilate5 = (8)*(np.random.rand(mask.shape[0],mask.shape[1])*1.0)+max_filter5  # 加深度 8mm 噪声
    depth_dilate5 = np.where(mask_dilate5, depth_dilate5, 0.0)
    depth_map = depth_map + depth_dilate5

    # 3. 头部 noise 大一些，主体部分 noise 小一些
    z = (depth_map)  # 空白区域深度为 0
    h, w = np.mgrid[1:mask.shape[0]+1,1:mask.shape[1]+1]
    x = (w-cx)*z/f
    y = (h-cy)*z/f
    distance = (x+chizhen['trans_x'])**2 + (y+chizhen['trans_y'])**2 + (z-chizhen['trans_z'])**2
    distance_thresh1 = 8**2

    noise1 = np.random.rand(mask.shape[0],mask.shape[1])*0.2-0.1
    noise1 = np.where(depth_map>0.01, noise1, 0.0)
    noise1 = np.where(distance<distance_thresh1, noise1, 0.0)
    noise2 = np.random.rand(mask.shape[0],mask.shape[1])-0.5 + np.random.rand()
    noise2 = np.where(depth_map>0.01, noise2, 0.0)
    noise2 = np.where(distance>=distance_thresh1, noise2, 0.0)
    depth_map = depth_map + noise1 + noise2

    # 4. 保存数据
    z = (depth_map)  # 空白区域深度为 0
    x = (w-cx)*z/f
    y = (h-cy)*z/f
    distance = (x+chizhen['trans_x'])**2 + (y+chizhen['trans_y'])**2 + (z-chizhen['trans_z'])**2
    distance_thresh3 = 7**2
    xyz = np.dstack((x,y,z)).reshape(-1,3)  # (HW,3)
    idx = np.where((xyz[:,2]>0.01))
    xyz = xyz[idx]  # (npoint,3)

    mask_pos = np.where(mask2>100, True, False)  # (H,W)
    mask_pos = np.where(distance<distance_thresh3, mask_pos, False).reshape(-1) # (HW,)
    mask_pos = mask_pos[idx]  # (npoint,)

    return xyz, mask_pos

def save_offset(bbox_writer, points_data, mask_pos, file_name):
    point_votes = np.ones((points_data.shape[0],4),dtype=np.float32)
    point_votes[:,1:4] = np.expand_dims(bbox_writer[0,0:3],0) - points_data[:,0:3]
    point_votes[np.where(mask_pos==False)] = 0
    np.savez_compressed(file_name, point_votes=point_votes)


if __name__ == '__main__':

    config_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),  # synthetic_data/
        "cfg/generate_mask_dataset.yaml"
    )
    # 转换为绝对路径
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)
    # open config file
    config = YamlConfig(config_filename)
    image_config = config["images"]

    # Create initial env to generate metadata
    env = BinHeapEnv(config)

    # data file
    render_dir = './output_chizhen_new'
    if not os.path.exists(render_dir + '/color'):
        os.mkdir(render_dir + '/color')
    if not os.path.exists(render_dir + '/depth'):
        os.mkdir(render_dir + '/depth')
    if not os.path.exists(render_dir + '/mask'):
        os.mkdir(render_dir + '/mask')
    if not os.path.exists(render_dir + '/pc'):
        os.mkdir(render_dir + '/pc')
    if 'shuangji' in tools:
        data_file_path = render_dir + "/shuangji.txt"
        shuangji_file = open(data_file_path, 'a')
    if 'zuzhi' in tools:
        data_file_path = render_dir + "/zuzhi.txt"
        zuzhi_file = open(data_file_path, 'a')
    if 'chizhen' in tools:
        data_file_path = render_dir + "/chizhen.txt"
        chizhen_file = open(data_file_path, 'a')

    i = 0
    num_states = 12000
    abandom = 0
    while i < num_states:

        index = i
        random.seed(index)

        if 'shuangji' in tools:
            tools['shuangji']['trans_x'] = ((random.random() - 0.5) * 40 - 30)
            tools['shuangji']['trans_y'] = ((random.random() - 0.5) * 40)
            tools['shuangji']['trans_z'] = ((random.random() - 0.5) * 40 + 50)
            tools['shuangji']['rot_x'] = (random.random() - 0.5) * 45 - 45
            tools['shuangji']['rot_y'] = (random.random() - 0.5) * 45
            tools['shuangji']['rot_z'] = (random.random() - 0.5) * 45 - 60
            tools['shuangji']['angle'] = (random.random()) * 80 - 40
            if tools['shuangji']['angle'] < 0:
                tools['shuangji']['angle'] = 0

            rot = [-tools['shuangji']['rot_x'], -tools['shuangji']
                        ['rot_y'], tools['shuangji']['rot_z']]
            while utils.data_filter_rotation(rot) == False:
                tools['shuangji']['rot_x'] = random.random() * 360
                tools['shuangji']['rot_y'] = random.random() * 360
                tools['shuangji']['rot_z'] = random.random() * 360
                rot = [-tools['shuangji']['rot_x'], -tools['shuangji']
                        ['rot_y'], tools['shuangji']['rot_z']]

        if 'zuzhi' in tools:
            tools['zuzhi']['trans_x'] = (random.random() - 0.5) * 40 + 30
            tools['zuzhi']['trans_y'] = (random.random() - 0.5) * 40
            tools['zuzhi']['trans_z'] = (random.random() - 0.5) * 40 + 50
            tools['zuzhi']['rot_x'] = (random.random() - 0.5) * 45 + 45
            tools['zuzhi']['rot_y'] = (random.random() - 0.5) * 45
            tools['zuzhi']['rot_z'] = (random.random() - 0.5) * 45 - 60
            tools['zuzhi']['angle'] = (random.random()) * 120 -60
            if tools['zuzhi']['angle'] < 0:
                tools['zuzhi']['angle'] = 0

            rot = [-tools['zuzhi']['rot_x'], -tools['zuzhi']
                        ['rot_y'], tools['zuzhi']['rot_z']]
            while utils.data_filter_rotation(rot) == False:
                tools['zuzhi']['rot_x'] = random.random() * 360
                tools['zuzhi']['rot_y'] = random.random() * 360
                tools['zuzhi']['rot_z'] = random.random() * 360
                rot = [-tools['zuzhi']['rot_x'], -tools['zuzhi']
                        ['rot_y'], tools['zuzhi']['rot_z']]

        if 'chizhen' in tools:
            tools['chizhen']['trans_z'] = random.random() * 70 + 20
            tools['chizhen']['trans_x'] = np.clip(random.gauss(0,0.5),-0.8,0.8) * tools['chizhen']['trans_z'] * 0.8
            tools['chizhen']['trans_y'] = np.clip(random.gauss(0,0.5),-0.8,0.8) * tools['chizhen']['trans_z'] * 0.4
            tools['chizhen']['rot_x'] = random.random() * 360
            tools['chizhen']['rot_y'] = random.random() * 360
            tools['chizhen']['rot_z'] = random.random() * 360
            tools['chizhen']['angle'] = (random.random()) * 60 - 30
            if tools['chizhen']['angle'] < 0:
                tools['chizhen']['angle'] = 0
            tools['chizhen']['theta_1'] = random.sample([0, 30, 60, 90], 1)[0]
            tools['chizhen']['rot_z_stem'] = random.random() * 360
            
            rot = [-tools['chizhen']['rot_x'], -tools['chizhen']
                        ['rot_y'], tools['chizhen']['rot_z']]
            while utils.data_filter_rotation(rot) == False:
                tools['chizhen']['rot_x'] = random.random() * 360
                tools['chizhen']['rot_y'] = random.random() * 360
                tools['chizhen']['rot_z'] = random.random() * 360
                rot = [-tools['chizhen']['rot_x'], -tools['chizhen']
                        ['rot_y'], tools['chizhen']['rot_z']]

            rot_stem = [0, 0, tools['chizhen']['rot_z_stem']]         
            R = np.matmul(utils.quat2R(utils.euler2quat(rot)), utils.quat2R(utils.euler2quat(rot_stem)))
            while utils.data_filter_stem_rotation(R) == False:
                tools['chizhen']['rot_z_stem'] = random.random() * 360            
                rot_stem = [0, 0, tools['chizhen']['rot_z_stem']]
                R = np.matmul(utils.quat2R(utils.euler2quat(rot_stem)), utils.quat2R(utils.euler2quat(rot)))

        # try:
        print("-------------Rendering cycle #" + str(index) + "-------------")
        # reset env
        env.reset(tools)
        state = env.state
        obs = env.render_camera_image(color=image_config["color"])
        if image_config["mask"]:
            # reset env
            env.reset(tools, is_mask=True, is_mask2=False)
            state = env.state
            obs2 = env.render_camera_image(color=False)
            depth_obs2 = obs2
            env.reset(tools, is_mask=True, is_mask2=True)
            state = env.state
            obs3 = env.render_camera_image(color=False)
            depth_obs3 = obs3
        if image_config["color"]:
            color_obs, depth_obs = obs  # depth_obs: np.array(np.float32)
        elif image_config["depth"]:
            depth_obs = obs
                
        # compute mask
        mask = 255*np.ones_like(depth_obs, dtype=np.uint8)
        mask2 = 255*np.ones_like(depth_obs, dtype=np.uint8)
        if image_config["mask"]:
            mask = np.where((np.abs(depth_obs2 - depth_obs) > 1e-6), mask, 0)
            mask2 = np.where((np.abs(depth_obs3 - depth_obs) > 1e-6), mask2, 0)

        # compute point cloud
        camera = config["state_space"]["camera"]
        camera_f = camera["focal_length"]
        points_data, mask_pos = depth2xyz(depth_obs, mask, mask2, tools['chizhen'],
            camera_f, float(camera["im_width"])/2.0, float(camera["im_height"])/2.0,
        )

        # 点太少了不要
        pcd_num = 4000
        if (points_data.shape[0])<pcd_num:
            # delete action objects
            for obj_state in state.obj_states:
                del obj_state
            del state
            abandom += 1
            i += 1
            continue
        else:
            choice = np.random.choice(points_data.shape[0], pcd_num, replace=False)
            points_data = points_data[choice, :]
            mask_pos = mask_pos[choice]

        # save results
        colorimg_filename = render_dir + '/color' + "/{:06d}.png".format(index)
        depthimg_filename = render_dir + '/depth' + "/{:06d}.png".format(index)
        mask_filename = render_dir + '/mask' + "/{:06d}.png".format(index)
        pc_filename = render_dir + '/pc' + "/{:06d}_pc.npz".format(index)
        # if image_config["color"]:
        #     ColorImage(color_obs).save(colorimg_filename)
        # if image_config["depth"]:
        #     DepthImage(depth_obs).save(depthimg_filename)
        # if image_config["mask"]:
        #     BinaryImage(mask2).save(mask_filename)
        np.savez_compressed(pc_filename, pc=points_data)

        bbox_writer = np.zeros((1,6),dtype=np.float32)
        T1 = RigidTransform(
            rotation=RigidTransform.z_axis_rotation((tools['chizhen']['rot_z']+180)*np.pi/180),
            translation=np.array([0.0,0.0,0.0]),
            from_frame="chizhen_1",
            to_frame="chizhen_0",
        )
        T2 = RigidTransform(
            rotation=RigidTransform.y_axis_rotation(tools['chizhen']['rot_y']*np.pi/180),
            translation=np.array([0.0,0.0,0.0]),
            from_frame="chizhen_2",
            to_frame="chizhen_1",
        )
        T3 = RigidTransform(
            rotation=RigidTransform.x_axis_rotation((tools['chizhen']['rot_x'])*np.pi/180),
            translation=np.array([0.0,0.0,0.0]),
            from_frame="chizhen_3",
            to_frame="chizhen_2",
        )
        T4 = RigidTransform(
            rotation=RigidTransform.z_axis_rotation((45)*np.pi/180),
            translation=np.array([0.0,0.0,0.0]),
            from_frame="intuitive",
            to_frame="chizhen_3",
        )
        T_intuitive = T1*T2*T3*T4
        R_intuitive = Rotation.from_matrix(T_intuitive.rotation)
        euler_intuitive = (R_intuitive.as_euler('XYZ'))  # rad
        bbox_writer[0,:] = [-tools['chizhen']['trans_x'], -tools['chizhen']['trans_y'], tools['chizhen']['trans_z'],
                            euler_intuitive[0], euler_intuitive[1], euler_intuitive[2]]  # euler
        np.save(render_dir + '/pc' + "/{:06d}_bbox.npy".format(index), bbox_writer)

        save_offset(bbox_writer, points_data, mask_pos, render_dir + '/pc' + "/{:06d}_votes.npz".format(index))

        # if 'shuangji' in tools:
        #     record_data(tools, 'shuangji', shuangji_file)
        # if 'zuzhi' in tools:
        #     record_data(tools, 'zuzhi', zuzhi_file)
        # if 'chizhen' in tools:
        #     record_data(tools, 'chizhen', chizhen_file)

        # def write_ply(points, filename):
        #     """ input: Nx3, write points to filename as PLY format. """
        #     points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
        #     vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
        #     el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        #     PlyData([el], text=True).write(filename)
        # write_ply(points_data, render_dir + '/pc' + "/{:06d}.ply".format(index))

        # delete action objects
        for obj_state in state.obj_states:
            del obj_state
        del state

        # update state id
        i += 1

        # except Exception as e:
        #     print('render fail')
        #     del env
        #     gc.collect()
        #     env = BinHeapEnv(config)
 
    # if 'shuangji' in tools:
    #     shuangji_file.close()
    # if 'zuzhi' in tools:
    #     zuzhi_file.close()
    # if 'chizhen' in tools:
    #     chizhen_file.close()
    print(abandom)
