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
  state_val_data = {}
  kinematics_data = {}
  seed = 0
  
  # Spatial reasoning task
  for _ in range(NUM_TESTS_PER_TASK):
    env = ClevrGridEnv(clevr_seed=seed, mujoco_seed=seed) 
    env.generate_llm_data(state_val_data, COLORS, DIRECT_COMB, DIRECTIONS)
    seed += 1
  db_utils.LLMDataset('state_validation_db', state_val_data)
    
  # Kinematics task
  for _ in range(NUM_TESTS_PER_TASK):
    env = ClevrGridEnv(clevr_seed=seed, mujoco_seed=seed)
    env.generate_llm_data(kinematics_data, COLORS, DIRECT_COMB, DIRECTIONS, kinematics=True)
    seed += 1
  db_utils.LLMDataset('kinematics_db', state_val_data)

if __name__ == '__main__':
  app.run(main)