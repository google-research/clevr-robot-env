# coding=utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

"""Generates descriptions based on a scene graph.

Adapted from the CLEVR dataset generation code.
"""
# pylint: skip-file
from __future__ import print_function

import json
import os
# import random
import re
import time

try:
  import third_party.clevr_robot_env_utils.question_engine as qeng
except ImportError as e:
  print(e)


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
synonyms_json = os.path.join(
    os.path.abspath(os.path.join(parent_dir, os.pardir)),
    'templates',
    'synonyms.json')


with open(synonyms_json, 'r') as f:
  SYNONYMS = json.load(f)


def generate_question_from_scene_struct(scene_struct,
                                        metadata,
                                        desc_templates,
                                        ques_templates,
                                        rng,
                                        description=True,
                                        instances_per_template=20,
                                        templates_per_image=3,
                                        verbose=False,
                                        time_dfs=False,
                                        use_synonyms=False):
  
  if description:
    templates = desc_templates
  else:
    templates = ques_templates
  
  questions = []
  templates_items = list(templates.items())
  num_instantiated = 0

  synonyms = SYNONYMS

  for (fn, idx), template in templates_items:
    ts, qs, ans = instantiate_templates_dfs(
        scene_struct,
        template,
        metadata,
        synonyms,
        rng,
        verbose=False,
        use_synonyms=use_synonyms)
    for t, q, a in zip(ts, qs, ans):
      questions.append({
          'question': t,
          'program': q,
          'answer': a,
          'template_filename': fn,
          'question_family_index': idx,
          'question_index': len(questions),
      })
    if ts:
      num_instantiated += 1
    if num_instantiated >= templates_per_image:
      break
  qa_pairs = []
  for q in questions:
    qa_pairs.append(' '.join([q['question'], str(q['answer'])]))
  return qa_pairs, questions

def instantiate_templates_dfs(scene_struct,
                              template,
                              metadata,
                              synonyms, rng,
                              verbose=False,
                              use_synonyms=True):
  param_name_to_type = {p['name']: p['type'] for p in template['params']}

  initial_state = {
      'nodes': [node_shallow_copy(template['nodes'][0])],
      'vals': {},
      'input_map': {
          0: 0
      },
      'next_template_node': 1,
  }
  states = [initial_state]
  relations = ["behind", "front", "left", "right"]
  final_states = []
  while states:
    state = states.pop()
    # Check to make sure the current state is valid
    q = {'nodes': state['nodes']}
    outputs = qeng.answer_question(q, metadata, scene_struct, all_outputs=True)
    answer = outputs[-1]
    if answer == '__INVALID__':
      continue

    # Check to make sure constraints are satisfied for the current state
    skip_state = False
    for constraint in template['constraints']:
      if constraint['type'] == 'NEQ':
        p1, p2 = constraint['params']
        v1, v2 = state['vals'].get(p1), state['vals'].get(p2)
        if v1 is not None and v2 is not None and v1 != v2:
          if verbose:
            print('skipping due to NEQ constraint')
            print(constraint)
            print(state['vals'])
          skip_state = True
          break
      elif constraint['type'] == 'EQ':
        p1, p2 = constraint['params']
        v1, v2 = state['vals'].get(p1), state['vals'].get(p2)
        if v1 is not None and v2 is not None and v1 == v2:
          if verbose:
            print('skipping due to EQ constraint')
            print(constraint)
            print(state['vals'])
          skip_state = True
          break
      elif constraint['type'] == 'NULL':
        p = constraint['params'][0]
        p_type = param_name_to_type[p]
        v = state['vals'].get(p)
        if v is not None:
          skip = False
          if p_type == 'Shape' and v != 'thing':
            skip = True
          if p_type != 'Shape' and v:
            skip = True
          if skip:
            if verbose:
              print('skipping due to NULL constraint')
              print(constraint)
              print(state['vals'])
            skip_state = True
            break
      elif constraint['type'] == 'COLOR':
        p = constraint['params'][0]
        p_type = param_name_to_type[p]
        v = state['vals'].get(p)
        cs = constraint['color']
        if p_type == 'Color' and p in state['vals']:
          eq = [v != c for c in cs]
          eq2 = True
          for e in eq:
            eq2 = eq2 and e
          if eq2:
            if verbose:  #  verbose
              print('skipping due to COLOR constraint')
              # print('{} is {} instead of {}'.format(p, v, c))
            skip_state = True
            break
      elif constraint['type'] == 'OUT_NEQ':
        i, j = constraint['params']
        i = state['input_map'].get(i, None)
        j = state['input_map'].get(j, None)
        if i is not None and j is not None and outputs[i] == outputs[j]:
          if verbose:
            print('skipping due to OUT_NEQ constraint')
            print(outputs[i])
            print(outputs[j])
          skip_state = True
          break
      elif constraint['type'] == 'SHAPE':
        p = constraint['params'][0]
        p_type = param_name_to_type[p]
        v = state['vals'].get(p)
        c = constraint['shape']
        if p_type == 'Shape' and p in state['vals'] and v != c:
          if verbose:  #  verbose
            print('skipping due to SHAPE constraint')
            print('{} is {} instead of {}'.format(p, v, c))
          skip_state = True
          break
      elif constraint['type'] == 'SIZE':
        p = constraint['params'][0]
        p_type = param_name_to_type[p]
        v = state['vals'].get(p)
        c = constraint['size']
        if p_type == 'Size' and p in state['vals'] and v != c:
          if verbose:  #  verbose
            print('skipping due to SIZE constraint')
            print('{} is {} instead of {}'.format(p, v, c))
          skip_state = True
          break
      elif constraint['type'] == 'MATERIAL':
        p = constraint['params'][0]
        p_type = param_name_to_type[p]
        v = state['vals'].get(p)
        c = constraint['material']
        if p_type == 'Material' and p in state['vals'] and v != c:
          if verbose:  #  verbose
            print('skipping due to MATERIAL constraint')
            print('{} is {} instead of {}'.format(p, v, c))
          skip_state = True
          break
      elif constraint['type'] == 'RELATION':
        p = constraint['params'][0]
        p_type = param_name_to_type[p]
        v = state['vals'].get(p)
        c = constraint['relation']
        if p_type == 'Relation' and p in state['vals'] and v != c:
          if verbose:  #  verbose
            print('skipping due to RELATION constraint')
            print('{} is {} instead of {}'.format(p, v, c))
          skip_state = True
          break
      elif constraint['type'] == 'COLOR_SET':
        p = constraint['params'][0]
        p_type = param_name_to_type[p]
        v = state['vals'].get(p)
        cs = constraint['color']
        if p_type == 'Color' and p in state['vals']:
          if v not in cs:
            if verbose:  #  verbose
              print('skipping due to COLOR_SET constraint')
              print('{} is {} instead of {}'.format(p, v, c))
            skip_state = True
            break
      elif constraint['type'] == 'SHAPE_SET':
        p = constraint['params'][0]
        p_type = param_name_to_type[p]
        v = state['vals'].get(p)
        cs = constraint['shape']
        if p_type == 'Shape' and p in state['vals']:
          if v not in cs:
            if verbose:  #  verbose
              print('skipping due to SHAPE_SET constraint')
              print('{} is {} instead of {}'.format(p, v, c))
            skip_state = True
            break
      elif constraint['type'] == 'SIZE_SET':
        p = constraint['params'][0]
        p_type = param_name_to_type[p]
        v = state['vals'].get(p)
        cs = constraint['size']
        if p_type == 'Size' and p in state['vals']:
          if v not in cs:
            if verbose:  #  verbose
              print('skipping due to SIZE_SET constraint')
              print('{} is {} instead of {}'.format(p, v, c))
            skip_state = True
            break
      elif constraint['type'] == 'MATERIAL_SET':
        p = constraint['params'][0]
        p_type = param_name_to_type[p]
        v = state['vals'].get(p)
        cs = constraint['material']
        if p_type == 'Material' and p in state['vals']:
          if v not in cs:
            if verbose:  #  verbose
              print('skipping due to MATERIAL_SET constraint')
              print('{} is {} instead of {}'.format(p, v, c))
            skip_state = True
            break
      else:
        assert False, 'Unrecognized constraint type "%s"' % constraint['type']

    if skip_state:
      continue

    # We have already checked to make sure the answer is valid, so if we have
    # processed all the nodes in the template then the current state is a valid
    # question, so add it if it passes our rejection sampling tests.
    if state['next_template_node'] == len(template['nodes']):
      # If the template contains a raw relate node then we need to check for
      # degeneracy at the end
      has_relate = any(n['type'] == 'relate' for n in template['nodes'])
      if has_relate:
        degen = qeng.is_degenerate(
            q, metadata, scene_struct, answer=answer, verbose=verbose)
        if degen:
          continue

      state['answer'] = answer
      if state['vals']['<R>'] in relations: 
        if state['answer'] == True:
          final_states.append(state)
        relations.remove(state['vals']['<R>'])
      if len(relations) == 0:
        break
      continue

    # Otherwise fetch the next node from the template
    # Make a shallow copy so cached _outputs don't leak ... this is very nasty
    next_node = template['nodes'][state['next_template_node']]
    next_node = node_shallow_copy(next_node)

    special_nodes = {
        'filter_unique',
        'filter_count',
        'filter_exist',
        'filter',
        'relate_filter',
        'relate_filter_unique',
        'relate_filter_count',
        'relate_filter_exist',
    }
    if next_node['type'] in special_nodes:
      # print('case 1 {}'.format(next_node['type']))
      if next_node['type'].startswith('relate_filter'):
        unique = (next_node['type'] == 'relate_filter_unique')
        include_zero = (
            next_node['type'] == 'relate_filter_count' or
            next_node['type'] == 'relate_filter_exist')
        filter_options = find_relate_filter_options(
            answer,
            scene_struct,
            metadata,
            rng,
            unique=unique,
            include_zero=include_zero)
      else:
        filter_options = find_filter_options(answer, scene_struct, metadata)
        if next_node['type'] == 'filter':
          # Remove null filter
          filter_options.pop((None, None, None, None), None)
        if next_node['type'] == 'filter_unique':
          # Get rid of all filter options that don't result in a single object
          filter_options = {
              k: v for k, v in filter_options.items() if len(v) == 1
          }
        else:
          # Add some filter options that do NOT correspond to the scene
          if next_node['type'] == 'filter_exist':
            # For filter_exist we want an equal number that do and don't
            num_to_add = len(filter_options)
          elif next_node['type'] == 'filter_count' or next_node[
              'type'] == 'filter':
            # For filter_count add nulls equal to the number of singletons
            num_to_add = sum(
                1 for k, v in filter_options.items() if len(v) == 1)
          add_empty_filter_options(filter_options, metadata, num_to_add, rng)

      filter_option_keys = list(filter_options.keys())
      rng.shuffle(filter_option_keys)
      for k in filter_option_keys:
        new_nodes = []
        cur_next_vals = {k: v for k, v in state['vals'].items()}
        next_input = state['input_map'][next_node['inputs'][0]]
        filter_side_inputs = next_node['side_inputs']
        if next_node['type'].startswith('relate'):
          param_name = next_node['side_inputs'][0]  # First one should be relate
          filter_side_inputs = next_node['side_inputs'][1:]
          param_type = param_name_to_type[param_name]
          assert param_type == 'Relation'
          param_val = k[0]
          k = k[1]
          # TODO: figure out what's wrong with this
          new_nodes.append({
              'type': 'relate',
              'inputs': [next_input - 1],
              'side_inputs': [param_val],
          })
          cur_next_vals[param_name] = param_val
          next_input = len(state['nodes']) + len(new_nodes) - 1
        for param_name, param_val in zip(filter_side_inputs, k):
          param_type = param_name_to_type[param_name]
          filter_type = 'filter_%s' % param_type.lower()
          if param_val is not None:
            new_nodes.append({
                'type': filter_type,
                'inputs': [next_input],
                'side_inputs': [param_val],
            })
            cur_next_vals[param_name] = param_val
            next_input = len(state['nodes']) + len(new_nodes) - 1
          elif param_val is None:
            if metadata['dataset'] == 'CLEVR-v1.0' and param_type == 'Shape':
              param_val = 'thing'
            else:
              param_val = ''
            cur_next_vals[param_name] = param_val
        input_map = {k: v for k, v in state['input_map'].items()}
        extra_type = None
        if next_node['type'].endswith('unique'):
          extra_type = 'unique'
        if next_node['type'].endswith('count'):
          extra_type = 'count'
        if next_node['type'].endswith('exist'):
          extra_type = 'exist'
        if extra_type is not None:
          new_nodes.append({
              'type': extra_type,
              'inputs': [input_map[next_node['inputs'][0]] + len(new_nodes)],
          })
        input_map[state['next_template_node']] = len(
            state['nodes']) + len(new_nodes) - 1
        states.append({
            'nodes': state['nodes'] + new_nodes,
            'vals': cur_next_vals,
            'input_map': input_map,
            'next_template_node': state['next_template_node'] + 1,
        })

    elif 'side_inputs' in next_node:
      print('case 2')
      # If the next node has template parameters, expand them out
      # TODO: Generalize to work for nodes with more than 1 side input
      assert len(next_node['side_inputs']) == 1, 'NOT IMPLEMENTED'

      # Use metadata to figure out domain of valid values for this parameter.
      # Iterate over the values in a random order; then it is safe to bail
      # from the DFS as soon as we find the desired number of valid template
      # instantiations.
      param_name = next_node['side_inputs'][0]
      param_type = param_name_to_type[param_name]
      param_vals = metadata['types'][param_type][:]
      rng.shuffle(param_vals)
      for val in param_vals:
        input_map = {k: v for k, v in state['input_map'].items()}
        input_map[state['next_template_node']] = len(state['nodes'])
        cur_next_node = {
            'type': next_node['type'],
            'inputs': [input_map[idx] for idx in next_node['inputs']],
            'side_inputs': [val],
        }
        cur_next_vals = {k: v for k, v in state['vals'].items()}
        cur_next_vals[param_name] = val

        states.append({
            'nodes': state['nodes'] + [cur_next_node],
            'vals': cur_next_vals,
            'input_map': input_map,
            'next_template_node': state['next_template_node'] + 1,
        })
    else:
      print('case 3')
      input_map = {k: v for k, v in state['input_map'].items()}
      input_map[state['next_template_node']] = len(state['nodes'])
      next_node = {
          'type': next_node['type'],
          'inputs': [input_map[idx] for idx in next_node['inputs']],
      }
      states.append({
          'nodes': state['nodes'] + [next_node],
          'vals': state['vals'],
          'input_map': input_map,
          'next_template_node': state['next_template_node'] + 1,
      })

  # Actually instantiate the template with the solutions we've found
  text_questions, structured_questions, answers = [], [], []
  for state in final_states:
    structured_questions.append(state['nodes'])
    answers.append(state['answer'])
    text = rng.choice(template['text'])
    for name, val in state['vals'].items():
      if val in synonyms and use_synonyms:
        val = rng.choice(synonyms[val])
      text = text.replace(name, val)
      text = ' '.join(text.split())
    text = replace_optionals(text, rng)
    text = ' '.join(text.split())
    text = other_heuristic(text, state['vals'])
    text_questions.append(text)

  return text_questions, structured_questions, answers


# =============================================================================
def replace_optionals(s, rng):
  """
  Each substring of s that is surrounded in square brackets is treated as
  optional and is removed with probability 0.5. For example the string

  "A [aa] B [bb]"

  could become any of

  "A aa B bb"
  "A  B bb"
  "A aa B "
  "A  B "

  with probability 1/4.
  """
  pat = re.compile(r'\[([^\[]*)\]')

  while True:
    match = re.search(pat, s)
    if not match:
      break
    i0 = match.start()
    i1 = match.end()
    if rng.random() > 0.5:
      s = s[:i0] + match.groups()[0] + s[i1:]
    else:
      s = s[:i0] + s[i1:]
  return s

def precompute_filter_options(scene_struct, metadata):
  # Keys are tuples (size, color, shape, material) (where some may be None)
  # and values are lists of object idxs that match the filter criterion
  attribute_map = {}

  if metadata['dataset'] == 'CLEVR-v1.0':
    attr_keys = ['size', 'color', 'material', 'shape']
  else:
    assert False, 'Unrecognized dataset'

  # Precompute masks
  masks = []
  for i in range(2**len(attr_keys)):
    mask = []
    for j in range(len(attr_keys)):
      mask.append((i // (2**j)) % 2)
    masks.append(mask)

  for object_idx, obj in enumerate(scene_struct['objects']):
    if metadata['dataset'] == 'CLEVR-v1.0':
      keys = [tuple(obj[k] for k in attr_keys)]

    for mask in masks:
      for key in keys:
        masked_key = []
        for a, b in zip(key, mask):
          if b == 1:
            masked_key.append(a)
          else:
            masked_key.append(None)
        masked_key = tuple(masked_key)
        if masked_key not in attribute_map:
          attribute_map[masked_key] = set()
        attribute_map[masked_key].add(object_idx)

  scene_struct['_filter_options'] = attribute_map


def find_filter_options(object_idxs, scene_struct, metadata):
  # Keys are tuples (size, color, shape, material) (where some may be None)
  # and values are lists of object idxs that match the filter criterion

  if '_filter_options' not in scene_struct:
    precompute_filter_options(scene_struct, metadata)

  attribute_map = {}
  object_idxs = set(object_idxs)
  for k, vs in scene_struct['_filter_options'].items():
    attribute_map[k] = sorted(list(object_idxs & vs))
  return attribute_map


def add_empty_filter_options(attribute_map, metadata, num_to_add, rng):
  # Add some filtering criterion that do NOT correspond to objects

  if metadata['dataset'] == 'CLEVR-v1.0':
    attr_keys = ['Size', 'Color', 'Material', 'Shape']
  else:
    assert False, 'Unrecognized dataset'

  attr_vals = [metadata['types'][t] + [None] for t in attr_keys]
  if '_filter_options' in metadata:
    attr_vals = metadata['_filter_options']

  target_size = len(attribute_map) + num_to_add
  while len(attribute_map) < target_size:
    k = (rng.choice(v) for v in attr_vals)
    if k not in attribute_map:
      attribute_map[k] = []


def find_relate_filter_options(object_idx,
                               scene_struct,
                               metadata, rng,
                               unique=False,
                               include_zero=False,
                               trivial_frac=0.1):
  options = {}
  if '_filter_options' not in scene_struct:
    precompute_filter_options(scene_struct, metadata)

  # TODO: Right now this is only looking for nontrivial combinations; in some
  # cases I may want to add trivial combinations, either where the intersection
  # is empty or where the intersection is equal to the filtering output.
  trivial_options = {}
  for relationship in scene_struct['relationships']:
    related = set(scene_struct['relationships'][relationship][object_idx])
    for filters, filtered in scene_struct['_filter_options'].items():
      intersection = related & filtered
      trivial = (intersection == filtered)
      if unique and len(intersection) != 1:
        continue
      if not include_zero and len(intersection) == 0:
        continue
      if trivial:
        trivial_options[(relationship, filters)] = sorted(list(intersection))
      else:
        options[(relationship, filters)] = sorted(list(intersection))

  N, f = len(options), trivial_frac
  num_trivial = int(round(N * f / (1 - f)))
  trivial_options = list(trivial_options.items())
  rng.shuffle(trivial_options)
  for k, v in trivial_options[:num_trivial]:
    options[k] = v

  return options


def node_shallow_copy(node):
  new_node = {
      'type': node['type'],
      'inputs': node['inputs'],
  }
  if 'side_inputs' in node:
    new_node['side_inputs'] = node['side_inputs']
  return new_node


def other_heuristic(text, param_vals):
  """Post-processing heuristic to handle the word "other"
  """
  if ' other ' not in text and ' another ' not in text:
    return text
  target_keys = {
      '<Z>',
      '<C>',
      '<M>',
      '<S>',
      '<Z2>',
      '<C2>',
      '<M2>',
      '<S2>',
  }
  if param_vals.keys() != target_keys:
    return text
  key_pairs = [
      ('<Z>', '<Z2>'),
      ('<C>', '<C2>'),
      ('<M>', '<M2>'),
      ('<S>', '<S2>'),
  ]
  remove_other = False
  for k1, k2 in key_pairs:
    v1 = param_vals.get(k1, None)
    v2 = param_vals.get(k2, None)
    if v1 != '' and v2 != '' and v1 != v2:
      print('other has got to go! %s = %s but %s = %s' % (k1, v1, k2, v2))
      remove_other = True
      break
  if remove_other:
    if ' other ' in text:
      text = text.replace(' other ', ' ')
    if ' another ' in text:
      text = text.replace(' another ', ' a ')
  return text
