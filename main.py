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

def record_data(tools: dict, file: str):
    data_file = open(file, 'w')
    for tool in tools:
        data_file.write(tool + '\n')
        data_file.write(str(tools[tool]['trans_x']) + '\n')
        data_file.write(str(tools[tool]['trans_y']) + '\n')
        data_file.write(str(tools[tool]['trans_z']) + '\n')
        data_file.write(str(tools[tool]['rot_x']) + '\n')
        data_file.write(str(tools[tool]['rot_y']) + '\n')
        data_file.write(str(tools[tool]['rot_z']) + '\n')
        data_file.write(str(tools[tool]['angle']) + '\n')

    data_file.close()


if __name__ == '__main__':

    offset = 0

    config_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),  # synthetic_data/
        "cfg/generate_mask_dataset.yaml"
    )
    # 转换为绝对路径
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)
    # open config file
    config = YamlConfig(config_filename)
    states_per_garbage_collect = config["states_per_garbage_collect"]  # 10
    image_config = config["images"]

    # Create initial env to generate metadata
    env = BinHeapEnv(config)

    i = 0
    num_states = 10
    while i < num_states:
        # sample states
        states_remaining = num_states - (i+offset)
        # Number of states before garbage collection (due to pybullet memory issues)
        for ii in range(min(states_per_garbage_collect, states_remaining)):  # (10, 100-state_id)

            index = i + offset
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
                tools['chizhen']['trans_z'] = random.random() * 90 + 10
                tools['chizhen']['trans_x'] = random.gauss(0, 0.5) * tools['chizhen']['trans_z'] * 0.8
                tools['chizhen']['trans_y'] = random.gauss(0, 0.5) * tools['chizhen']['trans_z'] * 0.4
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
                #print(R)
                while utils.data_filter_stem_rotation(R) == False:
                    tools['chizhen']['rot_z_stem'] = random.random() * 360            
                    rot_stem = [0, 0, tools['chizhen']['rot_z_stem']]
                    # print(rot_stem)
                    R = np.matmul(utils.quat2R(utils.euler2quat(rot_stem)), utils.quat2R(utils.euler2quat(rot)))


            render_dir = './output_shuangji_zuzhi_new/'

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
                color_obs2, depth_obs2 = obs2
            elif image_config["depth"]:
                depth_obs = obs
                depth_obs2 = obs2

            # compute mask
            if image_config["mask"]:
                mask = np.zeros_like(depth_obs, dtype=np.uint8)
                mask = np.where((np.abs(depth_obs2 - depth_obs) < 1e-6), mask, 255)

            # save results
            dir_name = render_dir + "/output_" + str(index)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            '''Generate rendered image'''
            colorimg_filename = dir_name + "colorimage_{:06d}.png".format(index)
            depthimg_filename = dir_name + "depthimage_{:06d}.png".format(index)
            mask_filename = dir_name + "mask_{:06d}.png".format(index)
            if image_config["color"]:
                ColorImage(color_obs).save(colorimg_filename)
            if image_config["depth"]:
                DepthImage(depth_obs).save(depthimg_filename)
            if image_config["mask"]:
                BinaryImage(mask).save(mask_filename)
            data_file_path = dir_name + "/data.txt"
            record_data(tools, data_file_path)

            # delete action objects
            for obj_state in state.obj_states:
                del obj_state
            del state
            gc.collect()

            # update state id
            i += 1

            # except Exception as e:
            #     print('render fail')
            #     del env
            #     gc.collect()
            #     env = BinHeapEnv(config)

        # garbage collect
        del env
        gc.collect()
        env = BinHeapEnv(config)

