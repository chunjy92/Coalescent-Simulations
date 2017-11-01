# -*- coding: utf-8 -*-

from typing import List

import numpy as np
from scipy.stats import poisson

from models import M, Sample

__author__ = 'Jayeol Chun'


DONE = []

def experiment(models, config):
  print("[*] Begin experimenting..")
  log_data = {}
  mu = config.mu
  mu_step = config.mu_step

  while mu < 5.0:
    print("Current Mu:", mu)
    apply_mutation(models, mu)
    mu = mu * mu_step
    flag = threshold(models, mu, log_data)
    if flag:
      print("Exiting due to flag")
      break
  return log_data


def threshold(models, mu, log_data):
  data = [model.bbl for model in models]
  keys = models[0].bbl.keys()

  for key in keys:
    if key in log_data.keys():
      # print("Skipping from inside threhold")
      continue
    print("Current Sample Size:", key)
    k, b = np.asarray(data[0][key]), np.asarray(data[1][key])
    kb_concat = np.concatenate([k,b])
    kb_label = np.concatenate([np.zeros_like(k), np.ones_like(b)])

    sort_idx = np.argsort(kb_concat)
    sorted_bbl = kb_concat[sort_idx]
    sorted_label = kb_label[sort_idx]

    threshold_bbl = threshold_idx = 0
    goodness = max_goodness = 0
    goodness_log = []

    for index, (bbl, label) in enumerate(zip(sorted_bbl, sorted_label)):
      goodness = goodness + 1 if label == 0 else goodness - 1
      goodness_log.append(goodness)
      if goodness > max_goodness:
        max_goodness = goodness
        threshold_idx = index
        try:
          threshold_bbl = (bbl + sorted_bbl[index+1]) / 2
        except IndexError:
          threshold_bbl = bbl

    pred_label = np.concatenate([np.zeros(threshold_idx+1), np.ones(len(sorted_bbl) - (threshold_idx + 1))])
    sum_correct = np.sum(pred_label == sorted_label)
    percent_correct = sum_correct / len(sorted_bbl)

    print("Threshold BBL:", threshold_bbl)
    print("Threshold IDX:", threshold_idx)
    print("Percent Correct:", percent_correct)
    print()

    if percent_correct >= .90:
      log_data[key] = (mu, threshold_bbl)
  return len(DONE) == len(keys)


def apply_mutation(models, mu):
  for model in models:
    for sample_size, trees in model.trees.items():
      if sample_size in DONE:
        continue
      bbl_list = list()
      for tree in trees:
        bbl = _apply_mutation(tree, mu)
        bbl_list.append(bbl)
      model.bbl[sample_size] = bbl_list


def _apply_mutation(tree, mu):
  nodes = collect_nodes(tree)
  bbl = 0
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
    else:
      bbl += node.mutations
  return bbl


def collect_nodes(sample):
  nodes = [sample]
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


def simulate(sample_size, model, num_iter, verbose=False):
  if verbose:
    print("\nSimjulating")
  trees = {}
  for _ in range(num_iter):
    coalescent_list = [Sample(s+1) for s in range(sample_size)]
    root = model.coalesce(coalescent_list, sample_size, verbose=verbose)
    # model.store_tree(root, key)
    if sample_size in trees:
      trees[sample_size].append(root)
    else:
      trees[sample_size] = [root]

  if verbose:
    print("Ended with {}, root identity -> {}".format(str(sample_size), root.identity))
  return trees


def _simulate(models: M, num_iter: int, sample_size: int, mu: float, data: List[np.ndarray],
              tree_dir:str=None, exp: bool=False, graphics: bool=False, verbose: bool=False):
  '''
  single simulation
  i.e. produces $num_iter many trees for each model
  and stores each statistics in to $data
  Returns some statistics, may change in the future
  '''
  # a_path = "/Users/jayeolchun/Documents/Research/Coalescent-Simulations/data"
  for i, model in enumerate(models):
    model = model(sample_size, mu)
    for iter in range(num_iter):
      coalescent_list = [Sample(s+1) for s in range(sample_size)]
      root = model.coalesce(coalescent_list, (iter, data[i]), exp=exp, verbose=verbose)
      model.store_tree(root)
      # if num_iter < 5 and graphics: display_tree(root, verbose=verbose)
    print(model.trees)
    model.save_trees(tree_dir)
  res = [(np.average(l), np.var(l)) for l in data]
  return res

def get_threshold(data, verbose=False):
  total_list = np.concatenate(data)
  total_size = len(total_list)
  total_label = np.concatenate([np.zeros_like(data[0]),np.ones_like(data[1])])
  result = np.concatenate([total_list, total_label], axis=1)
  sorted_result = result[np.argsort(result[:,0])]

  threshold_bbl = threshold_idx = 0
  goodness = max_goodness = 0
  goodness_log = []
  for index, res in enumerate(sorted_result):
    i = res[1]
    goodness = goodness+1 if i==0 else goodness-1
    goodness_log.append(goodness)
    if goodness > max_goodness:
      max_goodness = goodness
      threshold_idx = index
      try:
        threshold_bbl = (res[0]+sorted_result[index+1, 0]) / 2
      except IndexError:
        threshold_bbl = res[0]

  pred_label = np.concatenate([np.zeros(threshold_idx+1), np.ones(total_size-(threshold_idx+1))])
  sum_correct = np.sum(pred_label==sorted_result[:,1])
  percent_correct = sum_correct / total_size
  return threshold_bbl, percent_correct
