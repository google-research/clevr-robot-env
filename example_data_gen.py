"""This script demonstrate the usage of CLEVR-ROBOT environment."""

from __future__ import absolute_import, division, print_function
from absl import app, flags
from matplotlib import pyplot as PLT
from env import ClevrGridEnv

from utils import db_utils

FLAGS = flags.FLAGS
COLORS = ['red', 'blue', 'green', 'purple', 'cyan'] 
DIRECTIONS = ['West', 'East', 'South', 'North']
DIRECT_COMB = [('West', 'South'), ('West', 'North'), ('East', 'South'), ('East', 'North')]
NUM_TESTS_PER_TASK = 2

def main(_):
  data_dict = {}
  
  # Spatial reasoning task
  for _ in range(NUM_TESTS_PER_TASK):
    env = ClevrGridEnv(clevr_seed=len(data_dict), mujoco_seed=len(data_dict)) 
    env.generate_llm_data(data_dict, COLORS, DIRECT_COMB, DIRECTIONS)
    
  # Kinematics task
  for _ in range(NUM_TESTS_PER_TASK):
    env = ClevrGridEnv(clevr_seed=len(data_dict), mujoco_seed=len(data_dict))
    env.generate_llm_data(data_dict, COLORS, DIRECT_COMB, DIRECTIONS, kinematics=True)

  
  db_utils.LLMDataset('db', data_dict)
  
if __name__ == '__main__':
  app.run(main)