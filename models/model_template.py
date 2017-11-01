# -*- coding: utf-8 -*-

import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

# from simulation import display_tree
from .structure import Ancestor, T
from .utils import quicksort, update_children, display_tree

__author__ = 'Jayeol Chun'

TREE = ".tree"

class CoalescentModel(ABC):
  def __init__(self):
    self.identity = ""
    # $num_iter number of trees for each sample size in the sample_sizes list
    # {15: [root1, root2, ...], ...}
    self.trees = {}
    self.bbl   = {}


  @abstractmethod
  def _F(self, n: int, rate: np.ndarray = None):
    pass

  @abstractmethod
  def _coalesce_aux(self, coalescent_list) -> Tuple[float, int]:
    pass

  def save_trees(self, path):
    '''
    :param path: path to the folder where the model is to be saved
    '''
    print("Saving to", path, "for", self.identity)
    if not os.path.exists(path):
      os.makedirs(path, exist_ok=True)
    filename = self.identity + TREE
    out_path = os.path.join(path, filename)
    with open(out_path, 'wb') as f:
      pickle.dump(self.trees, f, protocol=pickle.HIGHEST_PROTOCOL)


  def _merge(self, coalescent_list: List[T], merge_identity: int, num_children: int=2) -> Tuple[Ancestor, List[T]]:
    '''
    performs a coalescent event that creates a merged ancestor
    -> returns the new ancestor and its children list
    '''
    # Create an Internal Node Representing a Coalescent Event
    merged_ancestor = Ancestor(merge_identity)

    # The merge_sample's immediate children chosen
    children = np.random.choice(coalescent_list, num_children, replace=False).tolist()

    # sort by big pivot for visual ease
    quicksort(children, 0, len(children)-1)
    coalescent_list.append(merged_ancestor)
    return merged_ancestor, children


  def coalesce(self, coalescent_list: List[T], sample_size: int, verbose: bool = False) -> Ancestor:
    generation_time = np.zeros(sample_size-1) # coalescent time for each generation, fixed: numpy array
    nth_coalescence = 0

    mean_pairwise_dist = 0.
    while len(coalescent_list) > 1:
      time, num_children = self._coalesce_aux(coalescent_list)
      generation_time[nth_coalescence] = time
      nth_coalescence +=1

      # merge children
      merged_ancestor, children = self._merge(coalescent_list, nth_coalescence, num_children)

      # update these children
      mean_pairwise_dist += update_children(merged_ancestor, children, sample_size, generation_time)

      for child in children:
        del coalescent_list[coalescent_list.index(child)]

    root = coalescent_list[0]
    root.identity = root.identity.replace('A', self.identity)
    root.pairwise_dist = mean_pairwise_dist
    if verbose:
      print("Coalescing Complete.")
      print("Total TEst Het:", mean_pairwise_dist)
      # print("Avg:", mean_pairwise_dist / (sample_size + num_coalescent_event))
    return root


  def plot_trees(self, verbose=False):
    print("Plotting..")
    for trees in self.trees.values():
      for tree in trees:
        display_tree(tree, verbose=verbose)
