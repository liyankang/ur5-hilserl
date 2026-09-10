'''
Author: liyankang 2638252904@qq.com
Date: 2025-03-27 13:00:35
LastEditors: liyankang 2638252904@qq.com
LastEditTime: 2025-11-18 00:37:29
FilePath: /hil-serl-sim_ur/env_test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# import debugpy
# debugpy.listen(10010)
# print('wait debugger')
# debugpy.wait_for_client()
# print("Debugger Attached")


from ur_sim.envs.panda_pick_gym_env import urPickCubeGymEnv
from ur_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from pynput import keyboard
from examples.experiments.pick_cube_sim.config import TrainConfig

# env = PandaPickCubeGymEnv(render_mode="human")
# env = PandaPickCubeGymEnv(render_mode="human", image_obs=True, config=EnvConfig())
env = TrainConfig().get_environment()
import numpy as np

obs, _ = env.reset()

while True:
    actions = env.action_space.sample()
    # actions = np.zeros(env.action_space.sample().shape) 
    next_obs, reward, done, truncated, info = env.step(actions)

    if done:
        obs, info = env.reset()
