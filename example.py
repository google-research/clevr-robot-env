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

FLAGS = flags.FLAGS

DIRECTIONS = [[1, 0], [0, 1], [-1, 0], [0, -1], [0.8, 0.8], [-0.8, 0.8], [0.8, -0.8], [-0.8, -0.8]]


def main(_):
  env = ClevrGridEnv()
  
  rgb = env.render(mode='rgb_array')
  PLT.imshow(rgb)
  PLT.savefig('init.jpeg')
  
  directions = [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
  velocities = [1, 0, 0, 0, 0]
  
  env.kinematics_step(directions, velocities, 1)
  
  rgb = env.render(mode='rgb_array')
  PLT.imshow(rgb)
  PLT.savefig('step_1.jpeg')
  
if __name__ == '__main__':
  app.run(main)
