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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from matplotlib import pyplot as PLT

from env import ClevrGridEnv

import numpy as np
import os

FLAGS = flags.FLAGS

DIRECTION_COORDS = [[1, 0], [0, 1], [-1, 0], [0, -1], [0.8, 0.8], [-0.8, 0.8], [0.8, -0.8], [-0.8, -0.8]]
COLORS = ['red', 'blue'] # , 'green', 'purple', 'cyan'
DIRECT_COMB = [('left', 'front'), ('left', 'behind'), ('right', 'front'), ('right', 'behind')]
DIRECTIONS = ['left', 'right', 'front', 'behind']

def main(_):
  file_dir = os.path.abspath(os.path.dirname(__file__))
  env = ClevrGridEnv(num_object=2, description_template_path=os.path.join(
    file_dir, 'templates/description_distribution_collision.json'), question_template_path=os.path.join(
    file_dir, 'templates/question_distribution_collision.json'), collision=True)
  
  rgb = env.render(mode='rgb_array')
  PLT.imshow(rgb)
  PLT.savefig("initial.jpg")
  
  description, colors_leftout = env.get_formatted_description()
  print('Descriptions: ', description)
  
  print('Initial coordinates:\n', 'Red sphere:', env.scene_struct['objects'][0]['3d_coords'], '\n', 'Blue sphere:', env.scene_struct['objects'][1]['3d_coords'])
  
  dir_idx = np.random.randint(low=0, high=len(DIRECTION_COORDS))
  # object = np.random.randint(low=0, high=env.num_object)
  object = 0
  force = np.zeros(env.num_object * 2)
  # force[object * 2] = DIRECTION_COORDS[dir_idx][0]
  # force[object * 2 + 1] = DIRECTION_COORDS[dir_idx][1]
  force[object * 2] = 0
  force[object * 2 + 1] = -2
  
  action = [
    object,
    force
  ]
  
  env.step(action)
  
  print('Final coordinates:\n', 'Red sphere:', env.scene_struct['objects'][0]['3d_coords'], '\n', 'Blue sphere:', env.scene_struct['objects'][1]['3d_coords'])
  
  rgb = env.render(mode='rgb_array')
  PLT.imshow(rgb)
  PLT.savefig("final.jpg")

if __name__ == '__main__':
  app.run(main)
