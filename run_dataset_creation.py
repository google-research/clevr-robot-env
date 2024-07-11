

from __future__ import absolute_import, division, print_function
from matplotlib import pyplot as PLT
from env import ClevrGridEnv

import tasks 
from utils import db_utils

COLORS = ['red', 'blue', 'green', 'purple', 'cyan'] 
DIRECTIONS = ['West', 'East', 'South', 'North']
DIRECT_COMB = [('West', 'South'), ('West', 'North'), ('East', 'South'), ('East', 'North')]
NUM_SCENARIOS_PER_TASK = 2

'''
This script creates our full tasks datasets in the datasets folder.
'''

def main():
    state_val_data = {}
    kinematics_data = {}

    # State validation task
    state_val_data = tasks.state_validation_task(NUM_SCENARIOS_PER_TASK, state_val_data, COLORS, DIRECT_COMB, DIRECTIONS)
    db_utils.create_db('datasets/state_validation_db', state_val_data, force_rewrite=False) 

    # State after Action task
    # TODO: integrate with the new code style

    # Kinematics task
    kinematics_data = tasks.kinematics_task(NUM_SCENARIOS_PER_TASK, kinematics_data, COLORS, DIRECT_COMB, DIRECTIONS)
    db_utils.create_db('datasets/kinematics_db', kinematics_data, force_rewrite=False)
    
  

if __name__ == '__main__':
    main()