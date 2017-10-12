import os
import pickle
from functools import partial
from multiprocessing import Pool
from typing import List
from scipy.stats import poisson

from models import MODELS
from simulation import simulate

__author__ = 'Jayeol Chun'

TREE = '.tree'


def threshold(models):
  for model in models:
    for trees in model.trees.values():
      bbl = get_bbl(trees)


def get_bbl(trees):
  for tree in trees:

  pass


def apply_mutation(models, mu):
  for model in models:
    for trees in model.trees.values():
      for tree in trees:
        _apply_mutation(tree, mu)

def _apply_mutation(tree, mu):
  nodes = collect_nodes(tree)
  for node in reversed(nodes):
    node.mutations = poisson.rvs(mu * node.time)
    if not node.is_sample():
      for child in node.children_list:
        if not child.is_sample() and child.mutations==0:
          idx = node.children_list.index(child)

          # delete 0 mutation ancestor
          del node.children_list[idx]

          # replace with its children list
          node.children_list[idx:idx] = child.children_list

          node.right = node.children_list[len(node.children_list)-1]
          node.big_pivot = node.right.big_pivot
          node.left = node.children_list[0]

          for i in reversed(range(1, len(node.children_list))):
            node.children_list[i].next = node.children_list[i-1]


def collect_nodes(sample):
  nodes = [sample]
  bbl = 0.
  current = sample.right
  _recurse(current, nodes)
  while current.next != sample.left:
    current = current.next
    _recurse(current, nodes)
  current = sample.left
  _recurse(current, nodes)
  return nodes

def _recurse(sample, nodes):
  nodes.append(sample)
  if sample.is_sample():
    return
  current = sample.right
  _recurse(current, nodes)
  while current.next != sample.left:
    current = current.next
    _recurse(current, nodes)
  current = sample.left
  _recurse(current, nodes)


def load_trees(input_dir:str) -> None:
  """Load Time trees and attach them to respective model objects"""
  assert os.path.exists(input_dir), "Error: Path to the trees does not exist.\nExiting.."
  print("Loading Time Trees..")
  models = [model() for model in MODELS] # init beforehand for classification of tree files
  for file in os.listdir(input_dir):
    if file.endswith(TREE):
      with open(os.path.join(input_dir, file), 'rb') as f:
        filename = file[:file.index('.')]
        model = models[0] if models[0].identity.lower().startswith(filename.lower()) else models[1]
        model.trees = pickle.load(f)
  return models

def generate_trees(sample_sizes: List, num_iter: int, num_proc=1, output_dir=None, verbose=False) -> dict:
  print("Generating Time Trees..")
  # out = {}
  models = []
  if num_proc > 1:
    print("Multi-Processing with {} Processes\n".format(num_proc))
    for model in MODELS:
      model = model()
      sim_wrapper = partial(simulate, model=model, num_iter=num_iter)
      result = {}
      with Pool(processes=num_proc) as pool:
        res = pool.map_async(sim_wrapper, sample_sizes)
        pool.close()
        pool.join()
      for tree in res.get():
        result.update(tree)
      model.trees = result
      if output_dir:
        model.save_trees(output_dir)
      # out[model.identity] = result
      models.append(model)

  else:
    print("Single-Processing\n")
    for model in MODELS:
      model = model()
      result = {}
      for sample_size in sample_sizes:
        trees = simulate(sample_size, model, num_iter, verbose=verbose)
        result.update(trees)
      model.trees = result
      if output_dir:
        model.save_trees(output_dir)
      # out[model.identity] =result
      models.append(model)
  return models
