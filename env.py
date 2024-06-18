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

"""The CLEVR-ROBOT environment."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random
import re
import itertools

from gym import spaces
from gym import utils
import numpy as np

from third_party.clevr_robot_env_utils.generate_question import generate_question_from_scene_struct
import third_party.clevr_robot_env_utils.generate_scene as gs
import third_party.clevr_robot_env_utils.question_engine as qeng

from utils import load_utils
from utils.xml_utils import convert_scene_to_xml

import cv2
import mujoco_env as mujoco_env  # custom mujoco_env
from dm_control import mujoco

file_dir = os.path.abspath(os.path.dirname(__file__))

DEFAULT_XML_PATH = os.path.join(file_dir, 'assets', 'clevr_default.xml')
FIXED_PATH = os.path.join(file_dir, 'templates', '10_fixed_objective.pkl')

# metadata
DEFAULT_METADATA_PATH = os.path.join(file_dir, 'metadata', 'metadata.json')
VARIABLE_OBJ_METADATA_PATH = os.path.join(file_dir, 'metadata',
                                          'variable_obj_meta_data.json')

# template_path
DESCRIPTION_DIST_TEMPLATE = os.path.join(
    file_dir, 'templates/description_distribution.json')
QUESTION_DIST_TEMPLATE = os.path.join(
    file_dir, 'templates/general_question_distribution.json')
VARIABLE_OBJ_TEMPLATE = os.path.join(file_dir, 'templates',
                                     'variable_object.json')


# fixed discrete action set
DIRECTIONS = [[1, 0], [0, 1], [-1, 0], [0, -1], [0.8, 0.8], [-0.8, 0.8],
              [0.8, -0.8], [-0.8, -0.8]]
X_RANGE, Y_RANGE = 0.7, 0.35


def _create_discrete_action_set():
  discrete_action_set = []
  for d in DIRECTIONS:
    for x in [-X_RANGE + i * X_RANGE / 5. for i in range(10)]:
      for y in [-Y_RANGE + i * 0.12 for i in range(10)]:
        discrete_action_set.append([[x, y], d])
  return discrete_action_set


DISCRETE_ACTION_SET = _create_discrete_action_set()

# cardinal vectors
# TODO: ideally this should be packaged into scene struct
four_cardinal_vectors = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
four_cardinal_vectors = np.array(four_cardinal_vectors, dtype=np.float32)
four_cardinal_vectors_names = ['front', 'behind', 'left', 'right']


class ClevrGridEnv(mujoco_env.MujocoEnv, utils.EzPickle):
  
  def __init__(self, num_object=5, description_template_path=None, question_template_path=None, collision=False, top_down_view=False):

    utils.EzPickle.__init__(self)
    self.min_dist = -0.5
    self.max_dist = 0.5
    self.curr_step = 0
    metadata_path=None
    self.frame_skip = 20
    self.reward_threshold = 0.
    question_template_path=None
    description_num=15
    self.maximum_episode_steps=100
    self.initial_xml_path = DEFAULT_XML_PATH
    self.obj_name = []
    self.action_type = 'perfect'
    self.use_movement_bonus = False
    self.direct_obs = False
    self.obs_type = 'direct'
    self.num_object = num_object
    self.variable_scene_content = False
    self.cache_valid_questions = False
    self.checker_board = False
    self.reward_scale = 1.0
    self.shape_val = 0.25
    self.min_move_dist = 0.05
    self.res = 64
    self.use_synonyms = False
    self.min_change_th = 0.26
    self.use_polar = False
    self.suppress_other_movement = False
    self.obj_radius = 0.1
    self.grid_placement_directions = {"right": (self.obj_radius*2.0, 0),
                   "left": (self.obj_radius*(-2.0), 0),
                   "front": (0, self.obj_radius*2.0),
                   "behind": (0, self.obj_radius*(-2.0))}

    train, test = load_utils.load_all_question(), None

    self.all_questions, self.held_out_questions = train, test
    self.all_question_num = len(self.all_questions)

    # loading meta data
    if metadata_path is None:
      metadata_path = DEFAULT_METADATA_PATH

    if self.variable_scene_content:
      print('loading variable input metadata')
      metadata_path = VARIABLE_OBJ_METADATA_PATH

    with open(metadata_path, 'r') as metadata_file:
      self.clevr_metadata = json.load(metadata_file)

    functions_by_name = {}
    for func in self.clevr_metadata['functions']:
      functions_by_name[func['name']] = func
    self.clevr_metadata['_functions_by_name'] = functions_by_name

    # information regarding question template
    if description_template_path is None:
      description_template_path = DESCRIPTION_DIST_TEMPLATE
    if question_template_path is None:
      question_template_path = QUESTION_DIST_TEMPLATE

    self.desc_template_num = 0
    self.desc_templates = {}
    fn = 'description_template'
    with open(description_template_path, 'r') as template_file:
      for i, template in enumerate(json.load(template_file)):
        self.desc_template_num += 1
        key = (fn, i)
        self.desc_templates[key] = template
    print('Read {} templates from disk'.format(self.desc_template_num))
    
    self.ques_template_num = 0
    self.ques_templates = {}
    fn = 'general_template'
    with open(question_template_path, 'r') as template_file:
      for i, template in enumerate(json.load(template_file)):
        self.ques_template_num += 1
        key = (fn, i)
        self.ques_templates[key] = template
    print('Read {} templates from disk'.format(self.ques_template_num))

    # setting up camera transformation
    self.w2c, self.c2w = gs.camera_transformation_from_pose(90, -45)

    # sample a random scene and struct
    self.scene_graph, self.scene_struct = self.sample_random_scene(collision=collision)

    # generate initial set of description from the scene graph
    self.description_num = description_num
    self.descriptions, self.full_descriptions = None, None
    self._update_description()
    self.obj_description = []
    self._update_object_description()
    obj_pos = [object["3d_coords"] for object in self.scene_struct["objects"]]

    mujoco_env.MujocoEnv.__init__(
        self,
        self.initial_xml_path,
        self.frame_skip,
        max_episode_steps=self.maximum_episode_steps,
        reward_threshold=self.reward_threshold,
        object_positions=None
    )

    # name of geometries in the scene
    self.obj_name = ['obj{}'.format(i) for i in range(self.num_object)]

    self.discrete_action_set = DISCRETE_ACTION_SET
    self.perfect_action_set = []
    for i in range(self.num_object):
      for d in DIRECTIONS:
        self.perfect_action_set.append(np.array([i] + d))


    self.action_space = spaces.Box(
          low=-1.0, high=1.1, shape=[4], dtype=np.float32)

    # setup camera and observation space
    self.camera = mujoco.MovableCamera(self.physics, height=300, width=300)
    self._top_down_view = top_down_view
    if top_down_view:
      camera_pose = self.camera.get_pose()
      self.camera.set_pose(camera_pose.lookat, camera_pose.distance,
                           camera_pose.azimuth, -90)
    self.camera_setup()

    self.observation_space = spaces.Box(
          low=0, high=255, shape=(self.res, self.res, 3), dtype=np.uint8)

    # agent type and randomness of starting location
    self.agent_type = 'pusher'
    self.random_start = False

    if not self.random_start:
      curr_scene_xml = convert_scene_to_xml(
          self.scene_graph,
          agent=self.agent_type,
          checker_board=self.checker_board)

    self.load_xml_string(curr_scene_xml)
    print('CLEVR-ROBOT environment initialized.')
    
  def kinematics_step(self, directions, velocities, time):
    new_coords_list = []
        
    for obj, direction, velocity in zip(self.scene_struct["objects"], directions, velocities):
      init_coords = obj["3d_coords"]
      displacement = [d * velocity * time for d in direction]
      new_coords = [init + disp for init, disp in zip(init_coords, displacement)]
      new_coords_list.append(new_coords)
            
    self.scene_graph, self.scene_struct = gs.generate_scene_struct(self.c2w, self.min_dist, self.max_dist, self.num_object, new_coords_list)
    self.scene_struct['relationships'] = gs.compute_relationship(self.scene_struct, use_polar=self.use_polar)
    self._update_description()
    self.curr_step += 1
    obj_pos = [item['3d_coords'] for item in self.scene_graph]
    mujoco_env.MujocoEnv.__init__(
        self,
        DEFAULT_XML_PATH,
        20,
        max_episode_steps=100,
        reward_threshold=0.,
        object_positions=obj_pos
    )
    

  def load_xml_string(self, xml_string):
    """Load the model into physics specified by a xml string."""
    self.physics.reload_from_xml_string(xml_string)

  def step_place_object_in_relation(self, obj1_id, relation, obj2_id, units=1):
    
    # get current object positions
    obj1_x, obj1_y, obj1_z = self.scene_graph[obj1_id]['3d_coords']
    obj2_x, obj2_y, obj2_z = self.scene_graph[obj2_id]['3d_coords']
    
    # compute desired position
    if(not (relation in self.grid_placement_directions)):
      raise NotImplementedError("Relation specified for place object step is not implemented!")
    
    obj1_x = obj2_x + self.grid_placement_directions[relation][0]
    obj1_y = obj2_y + self.grid_placement_directions[relation][1]
    
    # check that position is not occupied by another object
    positions = [obj['3d_coords'] for obj in self.scene_graph]
    
    # indicator of success in placement
    if (gs.no_overlap(obj1_x, obj1_y, positions, self.obj_radius) and gs.is_within_bounds(obj1_x, obj1_y, self.min_dist, self.max_dist)):
      self.scene_graph[obj1_id]['3d_coords'] = (obj1_x, obj1_y, obj1_z)
      self.scene_struct['objects'] = self.scene_graph
      self.scene_struct['relationships'] = gs.compute_relationship(
          self.scene_struct, use_polar=self.use_polar)
      
      # Need descriptions to be updated
      self._update_description()
      positions = [obj['3d_coords'] for obj in self.scene_graph]
      #TODO: any other way than going through init? Seems a bit roundabout.
      mujoco_env.MujocoEnv.__init__(
          self,
          self.initial_xml_path,
          self.frame_skip,
          max_episode_steps=self.max_episode_steps,
          reward_threshold=self.reward_threshold,
          object_positions=positions
      )
      return True
    else:
      return False
 
  def step(self,
           a,
           record_achieved_goal=False,
           goal=None,
           atomic_goal=False,
           update_des=False):
    """Take step a in the environment."""

    info = {}

    if not self.obj_name:
      self.do_simulation([0, 0], self.frame_skip)
      return self.get_obs(), 0, False, None

    # record questions that are currently false for relabeling
    currently_false = []
    if record_achieved_goal:
      if not self.cache_valid_questions:
        candidates = self.all_questions
      else:
        candidates = self.valid_questions
      random.shuffle(candidates)
      false_question_count = 0

      for q, p in candidates:
        if false_question_count > 128 and self.cache_valid_questions:
          break
        full_answer = self.answer_question(p, True)
        fixed_object_idx, fixed_object_loc = self._get_fixed_object(full_answer)
        if not full_answer[-1] and fixed_object_loc is not None:
          currently_false.append((q, p, fixed_object_idx, fixed_object_loc))
          false_question_count += 1

      random.shuffle(currently_false)
    
    self.step_discrete(a)

    self.curr_step += 1
    self._update_scene()
    if update_des:
      self._update_description()
      info['descriptions'] = self.descriptions
      info['full_descriptions'] = self.full_descriptions

    if record_achieved_goal:
      self.achieved_last_step = []
      self.achieved_last_step_program = []
      for q, p, obj_idx, obj_loc in currently_false:
        # fixed_object_idx
        obj_cur_loc = np.array(self.scene_graph[obj_idx]['3d_coords'])[:-1]
        # checking the first object has not been moved
        dispalcement = np.linalg.norm(obj_cur_loc - obj_loc)
        if self.answer_question(p) and dispalcement < self.min_change_th:
          self.achieved_last_step.append(q)
          self.achieved_last_step_program.append(p)

    done = self.curr_step >= self.max_episode_steps

    obs = self.get_obs()
    
    return obs, 0, done, info
  
  def _update_scene(self):
    """Update the scene description of the current scene."""
    self.previous_scene_graph = self.scene_graph
    for i, name in enumerate(self.obj_name):
      self.scene_graph[i]['3d_coords'] = tuple(self.get_body_com(name))
    self.scene_struct['objects'] = self.scene_graph
    self.scene_struct['relationships'] = gs.compute_relationship(
        self.scene_struct, use_polar=self.use_polar)
  
  def step_discrete(self, a):
    """Take discrete step by teleporting and then push."""
    self.do_simulation(list(np.array(a[1]) * 1.1), int(self.frame_skip * 6.0))
  
  def step_continuous(self, a):
    """Take a continuous version of step discrete."""
    a = np.squeeze(a)
    x, y, theta, r = a[0] * 0.7, a[1] * 0.7, a[2] * np.pi, a[3]
    direction = np.array([np.cos(theta), np.sin(theta)]) * 1.2
    duration = int((r + 1.0) * self.frame_skip * 3.0)
    new_loc = np.array([x, y])
    qpos, qvel = self.physics.data.qpos, self.physics.data.qvel
    qpos[-2:], qvel[-2:] = new_loc, np.zeros(2)
    self.set_state(qpos, qvel)
    curr_loc = self.get_body_com('point_mass')
    dist = [curr_loc - self.get_body_com(name) for name in self.obj_name]
    dist = np.min(np.linalg.norm(dist, axis=1))
    self.do_simulation(direction, duration)

  def answer_question(self, program, all_outputs=False):
    """Answer a functional program on the current scene."""
    return qeng.answer_question({'nodes': program},
                                self.clevr_metadata,
                                self.scene_struct,
                                cache_outputs=False,
                                all_outputs=all_outputs)

  def sample_random_scene(self, collision=False):
    """Sample a random scene base on current viewing angle."""
    if collision:
      return gs.generate_fixed_scene_struct(self.c2w, self.num_object,
                                      obj_pos=[[0, 0, 0.1], [0.2, 0.1, 0.1]])
    if self.variable_scene_content:
      return gs.generate_scene_struct(self.c2w, self.min_dist, self.max_dist, self.num_object,
                                      self.clevr_metadata)
    else:
      return gs.generate_scene_struct(self.c2w, self.min_dist, self.max_dist, self.num_object)
    
  def get_description(self):
    """Update and return the current scene description."""
    self._update_description()
    return self.descriptions, self.full_descriptions
  
  def get_program_from_question(self, question):
    program = [
        {'type': 'scene', 'inputs': []},
        {'type': 'filter_color', 'inputs': [0], 'side_inputs': []},
        {'type': 'filter_shape', 'inputs': [1], 'side_inputs': ['sphere']},
        {'type': 'exist', 'inputs': [2]},
        {'type': 'relate', 'inputs': [2], 'side_inputs': []},
        {'type': 'filter_color', 'inputs': [4], 'side_inputs': []},
        {'type': 'filter_shape', 'inputs': [5], 'side_inputs': ['sphere']},
        {'type': 'exist', 'inputs': [6]}
    ]
    
    parts = question.split(' ')
    main_color = parts[-2]
    direction = parts[5]
    second_color = parts[3]
    
    program[1]['side_inputs'].append(main_color)
    program[4]['side_inputs'].append(direction)
    program[5]['side_inputs'].append(second_color)

    return program
  
  def switch_behind_front(self, question):
    if 'behind' in question:
      return question.replace('behind', 'front')
    elif 'front' in question:
      return question.replace('front', 'behind')
    return question
  
  def format_questions(self, questions):
    formatted_questions = []

    for question_set in questions:
      if len(question_set) == 1:
        formatted_questions.append(self.switch_behind_front(question_set[0]))
      elif len(question_set) == 2:
        rel_1 = question_set[0].split(' ')
        rel_2 = question_set[1].split(' ')
        
        combined_question = f"Is there a {rel_1[3]} sphere {rel_1[5]} and {rel_2[5]} of the {rel_1[-2]} sphere?"
        formatted_questions.append(self.switch_behind_front(combined_question))
      else:
        formatted_questions.extend(question_set)

    return formatted_questions

  def generate_all_questions(self, colors, direction_combinations, directions):
    questions = []
    color_pairs = list(itertools.permutations(colors, 2))
    
    for direction in directions:
      for color1, color2 in color_pairs:
        questions.append([f"Is there a {color1} sphere {direction} of the {color2} sphere?"])
    
    for direction_combination in direction_combinations:
      for color1, color2 in color_pairs:
        question_set = [
          f"Is there a {color1} sphere {direction_combination[0]} of the {color2} sphere?",
          f"Is there a {color1} sphere {direction_combination[1]} of the {color2} sphere?"
        ]
        questions.append(question_set)
    
    return questions
  
  def get_ambiguous_pairs(self, description, colors):
    colors = ['red', 'blue', 'green', 'purple', 'cyan']
    all_combinations = itertools.combinations(colors, 2)
    all_combinations = set([tuple(sorted(combination)) for combination in all_combinations])
    
    pattern = re.compile(r'There is a (\w+) sphere.*?any (\w+) spheres')
    
    mentioned_combinations = set()
    for sentence in description:
      match = pattern.search(sentence)
      if match:
        color1 = match.group(1)
        color2 = match.group(2)
        mentioned_combinations.add(tuple(sorted([color1, color2])))
    
    not_mentioned_combinations = all_combinations - mentioned_combinations
    return list(not_mentioned_combinations)
  
  def generate_llm_questions(self, all_questions, colors_leftout):
    filtered_questions = []
    for idx, question in enumerate(all_questions):
      unk_answer = False 
      for color in colors_leftout:
        unk_answer = unk_answer or color[0] in question and color[1] in question
      if unk_answer:
        filtered_questions.append((question, idx))
    
    return filtered_questions
  
  def get_coordinates_description(self):
    objects = self.scene_struct['objects']
    
    color_order = ['red', 'blue', 'green', 'purple', 'cyan']
  
    coords = {}
    for idx, obj in enumerate(objects):
      coords[color_order[idx]] = obj['3d_coords']

    diameter = 0.2
    descriptions = []

    for i in range(1, len(color_order)):
      current_color = color_order[i]
      previous_color = color_order[i - 1]

      if current_color in coords and previous_color in coords:
        curr_coords = coords[current_color]
        prev_coords = coords[previous_color]

        x_diff = round((curr_coords[0] - prev_coords[0]) / diameter)
        y_diff = round((curr_coords[1] - prev_coords[1]) / diameter)
        
        description_parts = []
        if x_diff != 0:
          description_parts.append(f"{abs(x_diff)} unit{'s' if abs(x_diff) > 1 else ''} {'right of' if x_diff > 0 else 'left of'}")

        if y_diff != 0:
          description_parts.append(f"{abs(y_diff)} unit{'s' if abs(y_diff) > 1 else ''} {'behind' if y_diff > 0 else 'in front of'}")

        if description_parts:
          description = f"The {current_color} sphere is {' and '.join(description_parts)} the {previous_color} sphere."
          descriptions.append(description)

    return descriptions

  def get_formatted_description(self):
    """Get formatted decsription of the current scene for LLM input
    """
    unformatted, _ = self.get_description()
    
    def rephrase(sentence):
      # Extract the relevant parts using regex
      match = re.match(r'There is a (\w+) sphere[;,] are there any (\w+) spheres one unit (\w+) it\?', sentence)
      if match:
        main_color = match.group(1)
        other_color = match.group(2)
        position = match.group(3)
        # Switch "behind" and "front" and handle the "of"
        if position == "behind":
          return f'There is a {other_color} sphere one unit front of the {main_color} sphere'
        elif position == "front":
          return f'There is a {other_color} sphere one unit behind the {main_color} sphere'
        elif position == "left":
          return f'There is a {other_color} sphere one unit left of the {main_color} sphere'
        elif position == "right":
          return f'There is a {other_color} sphere one unit right of the {main_color} sphere'
        else:
          return f'There is a {other_color} sphere one unit {position} of the {main_color} sphere'
      return sentence

    # Rephrase the filtered data
    rephrased_data = [rephrase(item.split(' True')[0]) for item in unformatted]
    colors_leftout = self.get_ambiguous_pairs(unformatted, ['red', 'blue', 'green', 'purple', 'cyan'])
    
    return rephrased_data, colors_leftout

  def _update_description(self, custom_n=None):
    """Update the text description of the current scene."""
    gq = generate_question_from_scene_struct
    dn = self.description_num if not custom_n else custom_n
    tn = self.desc_template_num
    self.descriptions, self.full_descriptions = gq(
        self.scene_struct,
        self.clevr_metadata,
        self.desc_templates,
        self.ques_templates,
        description=True,
        templates_per_image=tn,
        instances_per_template=dn,
        use_synonyms=self.use_synonyms)

  def _update_object_description(self):
    """Update the scene description of the current scene."""
    self.obj_description = []
    for i in range(len(self.obj_name)):
      obj = self.scene_graph[i]
      color = obj['color']
      shape = obj['shape_name']
      material = obj['material']
      self.obj_description.append(' '.join([color, material, shape]))

  def get_obs(self):
    return self.get_image_obs()

  def get_image_obs(self):
    """Returns the image observation."""
    frame = self.render(mode='rgb_array')
    frame = cv2.resize(
        frame, dsize=(self.res, self.res), interpolation=cv2.INTER_CUBIC)
    return frame / 255.

  def reset(self, obj_pos=None):
    
    if(not(obj_pos is None)):
      """Reset with a fixed configuration."""

      self.scene_graph, self.scene_struct = gs.generate_fixed_scene_struct(self.c2w, self.num_object, obj_pos)

      # Generate initial set of description from the scene graph.
      self.descriptions, self.full_descriptions = None, None
      self._update_description()
      self.curr_step = 0

      curr_scene_xml = convert_scene_to_xml(
            self.scene_graph,
            agent=self.agent_type,
            checker_board=self.checker_board)

      self.load_xml_string(curr_scene_xml)

      self._update_object_description()

      return self.get_obs()
    else:
      
      raise NotImplementedError("NEED TO IMPLEMEN RANDOM RESET")
