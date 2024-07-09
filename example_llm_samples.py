# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script demonstrate the usage of CLEVR-ROBOT environment."""

from __future__ import absolute_import, division, print_function
from absl import app, flags
from matplotlib import pyplot as PLT
import tasks

FLAGS = flags.FLAGS
COLORS = ['red', 'blue', 'green', 'purple', 'cyan'] 
DIRECTIONS = ['West', 'East', 'South', 'North']
DIRECT_COMB = [('West', 'South'), ('West', 'North'), ('East', 'South'), ('East', 'North')]
NUM_TESTS_PER_TASK = 30

def main(_):
  data_dict = {}
  
  # State validation task
  data_dict = tasks.state_validation_task(NUM_TESTS_PER_TASK, data_dict, COLORS, DIRECT_COMB, DIRECTIONS)
    
  # Kinematics task
  data_dict = tasks.kinematics_task(NUM_TESTS_PER_TASK, data_dict, COLORS, DIRECT_COMB, DIRECTIONS)
  
if __name__ == '__main__':
  app.run(main)
