from env import ClevrGridEnv

def task_data_generation(num_scenes, data_dict, colors, direct_comb, directions, kinematics=False):
    for _ in range(num_scenes):
        env = ClevrGridEnv(clevr_seed=len(data_dict), mujoco_seed=len(data_dict))
        data_dict = env.generate_llm_data(data_dict, colors, direct_comb, directions, kinematics=kinematics)
    
    return data_dict
    