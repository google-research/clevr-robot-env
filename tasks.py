from env import ClevrGridEnv
from enum import Enum


def state_validation_task(num_scenes, data_dict, colors, direct_comb, directions):
    for _ in range(num_scenes):
        env = ClevrGridEnv(clevr_seed=len(data_dict), mujoco_seed=len(data_dict))
        data_dict = env.generate_llm_data(data_dict, colors, direct_comb, directions, step_type=None)
    return data_dict
    
def teleport_action_task(num_scenes, data_dict, colors, direct_comb, directions):
    for _ in range(num_scenes):
        env = ClevrGridEnv(clevr_seed=len(data_dict), mujoco_seed=len(data_dict))
        data_dict = env.generate_llm_data(data_dict, colors, direct_comb, directions, step_type="teleport")
    return data_dict
    
def kinematics_task(num_scenes, data_dict, colors, direct_comb, directions):
    for _ in range(num_scenes):
        env = ClevrGridEnv(clevr_seed=len(data_dict), mujoco_seed=len(data_dict))
        data_dict = env.generate_llm_data(data_dict, colors, direct_comb, directions, step_type="kinematic")
    return data_dict