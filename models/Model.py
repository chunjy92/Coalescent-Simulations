#! /usr/bin/python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

from .nodes import Ancestor
from utils import nC2, quicksort

import numpy as np

__author__ = 'Jayeol Chun'


class Model(ABC):
  """template model with behaviors in all children models"""
  def __init__(self, identity, name="Base Model"):
    self.identity = identity
    self.name = name

  def _merge(self, coalescent_list, merge_identity, num_children):
    '''
    performs a single coalescent event that creates a merged ancestor
    -> returns the new ancestor and its children list
    '''
    # Create an Internal Node Representing a Coalescent Event
    merged_ancestor = Ancestor(merge_identity)

    # The merge_sample's immediate children chosen
    children = np.random.choice(coalescent_list, num_children, replace=False).tolist()

    # Sort by big pivot for visual ease
    quicksort(children, 0, len(children)-1)
    coalescent_list.append(merged_ancestor)
    return merged_ancestor, children

  def _update_children(self, ancestor, children, sample_size, gen_time, verbose=False):
    '''
    for each child node under ancestor, calculate time and mutation value
    -> also sums each child's contribution to mean pairwise distance
    '''
    pairwise_dist = 0.
    for child in children:
      # _update_time(child, ancestor, gen_time)

      # 1. update time
      for j in range(ancestor.generation-1, child.generation-1, -1):
        child.time += gen_time[j]

      # 2. update heterozygosity contribution
      total_pair = nC2(sample_size)
      if child.is_sample():
        child.pairwise_dist = child.time * (sample_size-1) / total_pair
      else:
        num_child = len(child.descendent_list)
        child.pairwise_dist = child.time * (sample_size - num_child) * num_child / total_pair
      pairwise_dist += child.pairwise_dist
      # child.mutations = poisson.rvs(self.mu * child.time)

    # Update new information to the ancestor
    self._update_ancestor(ancestor, children)
    return pairwise_dist

  def _update_ancestor(self, ancestor, children_list):
    """
    assigns new attributes to the merged ancestor
    """
    ancestor.children_list = children_list
    ancestor.descendent_list = self._create_descendent_list(children_list)
    ancestor.right = children_list[len(children_list)-1]
    ancestor.big_pivot = ancestor.right.big_pivot
    ancestor.left = ancestor.children_list[0]

    # linked list, connecting each child nodes from right to left direction
    for i in reversed(range(1, len(children_list))):
      children_list[i].next = children_list[i - 1]

  def _create_descendent_list(self, children_list):
    """
    replaces children in the children_list with the child's own descendant list
    note that, by definition, `children_list' contain internal as well as leaf nodes one level below an ancestor,
    while `descendent_list' contain all leaf nodes anyway below it
    -> returns a new list with only Sample nodes
    """
    descendent_list = children_list[:]
    i = 0
    while i < len(descendent_list):
      if descendent_list[i].is_sample():
        i += 1
      else:
        size = len(descendent_list[i].descendent_list)
        descendent_list[i:i] = descendent_list[i].descendent_list
        del descendent_list[i+size]
        i += size
    return descendent_list

  def coalesce(self, coalescent_list, sample_size, verbose=False):
    """
    coalesces children until coalescent_list is empty
    -> returns the root, or the Most Recent Common Ancestor
    -> also calculates the mean pairwise distance for sanity check
    """
    generation_time = np.zeros(sample_size-1)
    nth_coalescence = 0
    mean_pairwise_dist = 0.

    while len(coalescent_list) > 1:
      time, num_children = self._coalesce(coalescent_list)
      generation_time[nth_coalescence] = time
      nth_coalescence += 1

      # merge children
      merged_ancestor, children = self._merge(coalescent_list, nth_coalescence, num_children)

      # update these children
      mean_pairwise_dist += self._update_children(merged_ancestor, children, sample_size, generation_time)

      for child in children:
        del coalescent_list[coalescent_list.index(child)]

    root = coalescent_list[0]
    root.identity = root.identity.replace('A', self.identity)
    root.pairwise_dist = mean_pairwise_dist
    if verbose:
      print("Coalescing Complete.")
      # print("Total TEst Het:", mean_pairwise_dist)
      # print("Avg:", mean_pairwise_dist / (sample_size + num_coalescent_event))
    return root

  def __repr__(self):
    return self.name

  @abstractmethod
  def F(self, n, rate):
    """
    characteristic function for each coalescent model
    :param n: sample size
    :param rate: for Bolthausen-Sznitman only
    :return * depends on the model
    """
    pass

  @abstractmethod
  def _coalesce(self, coalescent_list):
    """
    model-specific coalescent event simulation
    :param coalescent_list: List of nodes available for next coalescence
    :return (time of the coalescent event, number of children)ㅈ
    """
    pass
