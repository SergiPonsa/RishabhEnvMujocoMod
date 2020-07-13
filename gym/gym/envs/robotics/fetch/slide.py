import os
import numpy as np

from gym import utils
from gym.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'slide2.xml')


class FetchSlideEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot1:Actuator1': 0.0,
            'robot1:Actuator2': 0.392,
            'robot1:Actuator3': 0.0,
            'robot1:Actuator4': 1.962,
            'robot1:Actuator5': 0.0,
            'robot1:Actuator6': 0.78,
            'robot1:Actuator7': -1.57,
            'object0:joint': [1.7, 1.1, 0.41, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,\
            gripper_extra_height=-0.02, target_in_the_air=True, target_offset=0.0,\
            obj_range=0.15, target_range=0.3, distance_threshold=0.05,\
            initial_qpos=initial_qpos, reward_type=reward_type,object_pos_from_base=[0.4,0.0],\
            goal_pos_from_base=[0.5,0.2,0.0],joint_control=False)
        utils.EzPickle.__init__(self)
