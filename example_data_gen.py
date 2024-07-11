"""This script demonstrate the usage of CLEVR-ROBOT environment."""

from __future__ import absolute_import, division, print_function
from absl import app, flags
from matplotlib import pyplot as PLT
from env import ClevrGridEnv

import tasks 

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
  
  # State validation task
  state_val_data = tasks.state_validation_task(NUM_TESTS_PER_TASK, state_val_data, COLORS, DIRECT_COMB, DIRECTIONS)
  db_utils.LLMDataset('state_validation_db', state_val_data)
    
  # Kinematics task
  kinematics_data = tasks.kinematics_task(NUM_TESTS_PER_TASK, kinematics_data, COLORS, DIRECT_COMB, DIRECTIONS)
  db_utils.LLMDataset('kinematics_db', kinematics_data)

if __name__ == '__main__':
  app.run(main)