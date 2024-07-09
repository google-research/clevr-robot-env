# coding=utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

"""Generate scene descriptions.

Partially adapted from CLEVR dataset generation code.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np


def camera_transformation_from_pose(azimutal, elevation):
  """Output the camera transfromation matrix and inverse matrix."""
  azimutal, elevation = azimutal * 2. * np.pi / 360., elevation * 2. * np.pi / 360.
  azimutal *= -1.
  elevation *= -1.
  r_y = np.array([[np.cos(elevation), 0, np.sin(elevation)],
                  [0, 1, 0],
                  [-np.sin(elevation), 0, np.cos(elevation)]])
  r_z = np.array([[np.cos(azimutal), -np.sin(azimutal), 0],
                  [np.sin(azimutal), np.cos(azimutal), 0],
                  [0, 0, 1]])
  r = r_z.dot(r_y)
  # world_to_camera matrix, camera_to_world matrix
  return r, np.linalg.inv(r)


def generate_scene_struct(c2w, min_dist, max_dist, rng, num_object=3, coords=None):
  """Generate a random scene struct."""
  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': 'none',
      'objects': [],
      'directions': {},
  }

  plane_normal = np.array([0, 0, 1.])
  cam_behind = c2w.dot(np.array([-1., 0, 0]))
  cam_left = c2w.dot(np.array([0, 1., 0]))
  cam_up = c2w.dot(np.array([0, 0, 1.]))
  plane_behind = cam_behind - cam_behind.dot(plane_normal) * plane_normal
  plane_left = cam_left - cam_left.dot(plane_normal) * plane_normal
  plane_up = cam_up.dot(plane_normal) * plane_normal
  plane_behind /= np.linalg.norm(plane_behind)
  plane_left /= np.linalg.norm(plane_left)
  plane_up /= np.linalg.norm(plane_up)

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['South'] = plane_behind
  scene_struct['directions']['North'] = -plane_behind
  scene_struct['directions']['West'] = plane_left
  scene_struct['directions']['East'] = -plane_left
  scene_struct['directions']['above'] = plane_up
  scene_struct['directions']['below'] = -plane_up

  # Now make some random objects
  # objects = add_random_objects(scene_struct, num_object, metadata=metadata)
  objects = add_objects_grid(num_object, min_dist, max_dist, rng, coords)
  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_relationship(scene_struct)
  return objects, scene_struct

def no_overlap(new_x, new_y, positions, radius):
  for x, y, _ in positions:
    distance = math.sqrt((new_x - x) ** 2 + (new_y - y) ** 2)
    if distance < 2 * radius:
      return False
  return True

def is_within_bounds(x, y, min_dist, max_dist):
  return min_dist <= x <= max_dist and min_dist <= y <= max_dist

def add_objects_grid(num_objects, min_dist, max_dist, rng, coords=None, grid_obj_radius=0.1):
  positions = []
  objects = []
  
  # size_mapping = [('small', 0.07), ('medium', 0.1), ('large', 0.13)]
  size_mapping = [('large', grid_obj_radius)]
  # shape_mapping = [('sphere', 'sphere'), ('box', 'cube'),
  #                  ('cylinder', 'cylinder')]
  shape_mapping = [('sphere', 'sphere')]
  color_mapping = [('red', '1 0.1 0.1 1'), ('blue', '0.2 0.5 1 1'),
                    ('green', '0.2 1 0 1'), ('purple', '0.8 0.2 1 1'),
                    ('cyan', '0.2 1 1 1')]
  material_mapping = ['rubber']
  

  # Place the first object randomly within bounds
  # allow for spawning objects in a variety of 2 unit x and y directions. 
  list_grid_dirs = [(x, y) for x in [-grid_obj_radius*4.0, -grid_obj_radius*2.0, 0, grid_obj_radius*2.0, grid_obj_radius*4.0] for y in [-grid_obj_radius*4.0, -grid_obj_radius*2.0, 0, grid_obj_radius*2.0, grid_obj_radius*4.0] if not (x == 0 and y == 0)]
  
  if not coords:
    x = rng.uniform(min_dist, max_dist)
    y = rng.uniform(min_dist, max_dist)
  else:
    x = coords[0][0]
    y = coords[0][1]
  
  for i in range(num_objects):
    size_name, r = rng.choice(size_mapping)
    shape_name, shape = rng.choice(shape_mapping)
    color_name, color = color_mapping[i]
    mat_name = rng.choice(material_mapping)
    
    # First object
    if i == 0:
      positions.append((x, y, r))
      theta = 360.0 * rng.random()
      objects.append({
          'shape': shape,
          'shape_name': shape_name,
          'size': size_name,
          '3d_coords': (x, y, r),
          'color_val': color,
          'color': color_name,
          'rotation': theta,
          'material': mat_name,
      })
      continue
      
    last_x, last_y, _ = positions[-1]
    
    rng.shuffle(list_grid_dirs)

    if not coords:
      for dx, dy in list_grid_dirs:
        new_x = last_x + dx
        new_y = last_y + dy
        if is_within_bounds(new_x, new_y, min_dist + r, max_dist - r) and no_overlap(new_x, new_y, positions, size_mapping[0][1]):
          break
    else:
      new_x = coords[i][0]
      new_y = coords[i][1]
        
    positions.append((new_x, new_y, r))
    theta = 360.0 * rng.random()
    objects.append({
      'shape': shape,
      'shape_name': shape_name,
      'size': size_name,
      '3d_coords': (new_x, new_y, r),
      'color_val': color,
      'color': color_name,
      'rotation': theta,
      'material': mat_name,
    })

  return objects

def generate_fixed_scene_struct(c2w, num_object=3, obj_pos=[]):
  """Generate a fixed scene struct."""
  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': 'none',
      'objects': [],
      'directions': {},
  }

  plane_normal = np.array([0, 0, 1.])
  cam_behind = c2w.dot(np.array([-1., 0, 0]))
  cam_left = c2w.dot(np.array([0, 1., 0]))
  cam_up = c2w.dot(np.array([0, 0, 1.]))
  plane_behind = cam_behind - cam_behind.dot(plane_normal) * plane_normal
  plane_left = cam_left - cam_left.dot(plane_normal) * plane_normal
  plane_up = cam_up.dot(plane_normal) * plane_normal
  plane_behind /= np.linalg.norm(plane_behind)
  plane_left /= np.linalg.norm(plane_left)
  plane_up /= np.linalg.norm(plane_up)

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['South'] = plane_behind
  scene_struct['directions']['North'] = -plane_behind
  scene_struct['directions']['West'] = plane_left
  scene_struct['directions']['East'] = -plane_left
  scene_struct['directions']['above'] = plane_up
  scene_struct['directions']['below'] = -plane_up

  # Now make some random objects
  objects = add_fixed_objects(scene_struct, num_object, obj_pos)
  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_relationship(scene_struct)
  return objects, scene_struct

def add_fixed_objects(scene_struct,
                       num_objects,
                       obj_pos,
                       max_retries=10,
                       min_margin=0.01,
                       min_dist=0.1,
                       metadata=None):
  """Add fixed objects to scene struct."""
  positions = []
  objects = []

  color_mapping = [('red', '1 0.1 0.1 1'), ('blue', '0.2 0.5 1 1'),
                    ('green', '0.2 1 0 1'), ('purple', '0.8 0.2 1 1'),
                    ('cyan', '0.2 1 1 1')]
  position_mapping = obj_pos
  rotation_mapping = [0., 0., 0., 0., 0.]

  assert len(color_mapping) >= num_objects

  for i in range(num_objects):
    size_name = 'large'
    r = 0.13
    shape_name = 'sphere'
    shape = 'sphere'
    color_name, color = color_mapping[i]
    mat_name = 'rubber'
    positions.append(position_mapping[i])
    theta = rotation_mapping[i]
    
    objects.append({
        'shape': shape,
        'shape_name': shape_name,
        'size': size_name,
        '3d_coords': position_mapping[i],
        'color_val': color,
        'color': color_name,
        'rotation': theta,
        'material': mat_name,
    })
  return objects

def add_random_objects(scene_struct,
                       num_objects, rng,
                       max_retries=10,
                       min_margin=0.01,
                       min_dist=0.1,
                       metadata=None):
  """Add a random number of object to scene struct."""
  positions = []
  objects = []
  if not metadata:
    # size_mapping = [('small', 0.07), ('medium', 0.1), ('large', 0.13)]
    size_mapping = [('large', 0.13)]
    # shape_mapping = [('sphere', 'sphere'), ('box', 'cube'),
    #                  ('cylinder', 'cylinder')]
    shape_mapping = [('sphere', 'sphere')]
    color_mapping = [('red', '1 0.1 0.1 1'), ('blue', '0.2 0.5 1 1'),
                     ('green', '0.2 1 0 1'), ('purple', '0.8 0.2 1 1'),
                     ('cyan', '0.2 1 1 1')]
    material_mapping = ['rubber']
  else:
    size_mapping = metadata['types']['SizeValue']
    shape_mapping = metadata['types']['ShapeValue']
    color_mapping = metadata['types']['ColorValue']
    material_mapping = metadata['types']['Material']
    all_combination = []
    for si in size_mapping:
      for sh in shape_mapping:
        for c in color_mapping:
          for m in material_mapping:
            all_combination.append((si, sh, c, m))

  assert len(color_mapping) >= num_objects

  for i in range(num_objects):

    if not metadata:
      size_name, r = rng.choice(size_mapping)
      shape_name, shape = rng.choice(shape_mapping)
      if not metadata:
        color_name, color = color_mapping[i]
      else:
        color_name, color = rng.choice(color_mapping)
      mat_name = rng.choice(material_mapping)
    else:
      idx = rng.choice(list(range(len(all_combination))))
      size_tuple, shape_tuple, color_tuple, mat_tuple = all_combination.pop(idx)
      size_name, r = size_tuple
      shape_name, shape = shape_tuple
      color_name, color = color_tuple
      mat_name = mat_tuple

    num_tries = 0
    while True:
      num_tries += 1
      if num_tries > max_retries:
        return add_random_objects(scene_struct, num_objects, rng, metadata=metadata)
      x = rng.uniform(-0.5, 0.5)
      y = rng.uniform(-0.3, 0.5)
      dists_good, margins_good = True, True
      for (xx, yy, rr) in positions:
        dx, dy = x - xx, y - yy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist - r - rr < min_dist:
          dists_good = False
          break
        for direction_name in ['West', 'East', 'North', 'South']:
          direction_vec = scene_struct['directions'][direction_name]
          assert direction_vec[2] == 0
          margin = dx * direction_vec[0] + dy * direction_vec[1]
          if 0 < margin < min_margin:
            # print(margin, min_margin, direction_name)
            # print('BROKEN MARGIN!')
            margins_good = False
            break
        if not margins_good:
          break
      if dists_good and margins_good:
        break
    if shape_name == 'box':
      r /= math.sqrt(2) * 0.9
    if shape_name == 'cylinder':
      r /= 1.2

    positions.append((x, y, r))
    theta = 360.0 * rng.random()
    objects.append({
        'shape': shape,
        'shape_name': shape_name,
        'size': size_name,
        '3d_coords': (x, y, r),
        'color_val': color,
        'color': color_name,
        'rotation': theta,
        'material': mat_name,
    })
  return objects


def randomly_perturb_objects(scene_struct,
                             old_objects, rng,
                             max_retries=10,
                             min_margin=0.01,
                             min_dist=0.1):
  """Randomly perturb the scene struct's object without sampling new ones."""
  num_objects = len(old_objects)
  positions = []
  objects = []
  size_lookup = {'small': 0.07, 'medium': 0.1, 'large': 0.13}
  for i in range(num_objects):
    old_o = old_objects[i]
    size_name, r = old_o['size'], size_lookup[old_o['size']]
    shape_name, shape = old_o['shape_name'], old_o['shape']
    color_name, color = old_o['color'], old_o['color_val']
    mat_name = old_o['material']

    num_tries = 0
    while True:
      num_tries += 1
      if num_tries > max_retries:
        return randomly_perturb_objects(scene_struct, old_objects, rng)
      x = rng.uniform(-0.5, 0.5)
      y = rng.uniform(-0.3, 0.5)
      dists_good, margins_good = True, True
      for (xx, yy, rr) in positions:
        dx, dy = x - xx, y - yy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist - r - rr < min_dist:
          dists_good = False
          break
        for direction_name in ['left', 'right', 'front', 'behind']:
          direction_vec = scene_struct['directions'][direction_name]
          assert direction_vec[2] == 0
          margin = dx * direction_vec[0] + dy * direction_vec[1]
          if 0 < margin < min_margin:
            margins_good = False
            break
        if not margins_good:
          break
      if dists_good and margins_good:
        break

    positions.append((x, y, r))
    theta = 360.0 * rng.random()
    objects.append({
        'shape': shape,
        'shape_name': shape_name,
        'size': size_name,
        '3d_coords': (x, y, r),
        'color_val': color,
        'color': color_name,
        'rotation': theta,
        'material': mat_name,
    })
  return objects


def compute_relationship(scene_struct, use_polar=False, eps=0.001, max_dist=1):
  """Compute pariwise relationship between objects."""
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below':
      continue
    all_relationships[name] = []
    for _, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2:
          continue
        coords2 = obj2['3d_coords']
        diff = np.array([coords2[k] - coords1[k] for k in [0, 1, 2]])
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        qualified = dot > eps
        if qualified:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships
