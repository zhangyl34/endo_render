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

from plyfile import PlyData, PlyElement

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


def depth2xyz(depth_map, mask, fx, fy, cx, cy):
    
    h, w = np.mgrid[1:depth_map.shape[0]+1,1:depth_map.shape[1]+1]

    # put on the mask
    depth_map = np.where((mask>100), depth_map, 0)

    # mask dilate
    kernel = np.asarray([[0,255,0],[255,255,255],[0,255,0]])
    mask_dilate = convolve2d(mask.astype(int), kernel.astype(int), mode='same').astype(bool)
    mask_dilate = np.where((mask>100), False, mask_dilate)  # 只保留膨胀出来的区域为 True
    kernel = np.asarray([[0,1,0],[1,1,1],[0,1,0]])
    max_filter = signal.order_filter(depth_map, kernel, 4)  # 邻域最大值
    depth_dilate = (100.0-max_filter)*(np.random.rand(max_filter.shape[0],max_filter.shape[1])*2.0-1.0)+max_filter  # 加噪声
    depth_dilate = np.where(depth_dilate<max_filter, max_filter, depth_dilate)
    depth_dilate = np.where(mask_dilate, depth_dilate, 0)
    depth_map = depth_map + depth_dilate

    # add some noise.
    noise = 0.1  # -0.1~0.1 mm
    z = (depth_map - np.ones([depth_map.shape[0], depth_map.shape[1]])*noise +
        np.random.random([depth_map.shape[0], depth_map.shape[1]])*noise*2)  # 空白区域深度为 0
    x = (w-cx)*z/fx
    y = (h-cy)*z/fy
    xyz = np.dstack((x,y,z)).reshape(-1,3)

    # remove -depth point (empty points)
    thresh = noise*2
    xyz = xyz[~(xyz[:,2]<thresh)]
    
    return xyz

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
    render_dir = './output_chizhen'
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

    i = 100
    num_states = 1000
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
            env.reset(tools, is_mask=True)
            state = env.state
            obs2 = env.render_camera_image(color=image_config["color"])
        if image_config["color"]:
            color_obs, depth_obs = obs  # depth_obs: np.array(np.float32)
            if image_config["mask"]:
                color_obs2, depth_obs2 = obs2
        elif image_config["depth"]:
            depth_obs = obs
            if image_config["mask"]:
                depth_obs2 = obs2

        # compute mask
        mask = 255*np.ones_like(depth_obs, dtype=np.uint8)
        if image_config["mask"]:
            mask = np.where((np.abs(depth_obs2 - depth_obs) > 1e-6), mask, 0)

        # compute point cloud
        camera = config["state_space"]["camera"]
        camera_f = camera["focal_length"]
        points_data = depth2xyz(depth_obs, mask,
            camera_f, camera_f, float(camera["im_width"])/2.0,
            float(camera["im_height"])/2.0, 
        )

        # save results
        colorimg_filename = render_dir + '/color' + "/{:06d}.png".format(index)
        depthimg_filename = render_dir + '/depth' + "/{:06d}.png".format(index)
        mask_filename = render_dir + '/mask' + "/{:06d}.png".format(index)
        pc_filename = render_dir + '/pc' + "/{:06d}.npz".format(index)
        # if image_config["color"]:
        #     ColorImage(color_obs).save(colorimg_filename)
        # if image_config["depth"]:
        #     DepthImage(depth_obs).save(depthimg_filename)
        # if image_config["mask"]:
        #     BinaryImage(mask).save(mask_filename)
        np.savez_compressed(pc_filename, pc=points_data)
        
        if 'shuangji' in tools:
            record_data(tools, 'shuangji', shuangji_file)
        if 'zuzhi' in tools:
            record_data(tools, 'zuzhi', zuzhi_file)
        if 'chizhen' in tools:
            record_data(tools, 'chizhen', chizhen_file)

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
 
    if 'shuangji' in tools:
        shuangji_file.close()
    if 'zuzhi' in tools:
        zuzhi_file.close()
    if 'chizhen' in tools:
        chizhen_file.close()

