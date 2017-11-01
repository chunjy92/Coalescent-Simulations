# -*- coding: utf-8 -*-

import math
from typing import List
import numpy as np
from .structure import Sample, Ancestor, T

__author__ = 'Jayeol Chun'


def update_children(ancestor: Ancestor, children: List[T],
                    sample_size: int, gen_time: np.ndarray, verbose=False):
  '''
  for each child node under ancestor, calculate time and mutation value
  '''
  # BEGIN: iteration through the children_list
  pairwise_dist = 0.
  for child in children:
    _update_time(child, ancestor, gen_time)
    # update heterozygosity contribution
    total_pair = nC2(sample_size)
    if child.is_sample():
      child.pairwise_dist = child.time * (sample_size-1) / total_pair
    else:
      num_child = len(child.descendent_list)
      child.pairwise_dist = child.time * (sample_size-num_child) * num_child / total_pair
    pairwise_dist += child.pairwise_dist
    # child.mutations = poisson.rvs(self.mu * child.time)

  # Update new information to the ancestor
  update_ancestor(ancestor, children)
  return pairwise_dist

def _update_time(sample: T, ancestor: Ancestor, gen_time: np.ndarray):
  """
  adds up the between-generation time
  """
  for j in range(ancestor.generation-1, sample.generation-1, -1):
    sample.time += gen_time[j]


def update_ancestor(ancestor: Ancestor, children_list: List[T]):
  """
  assigns new attributes to the merged ancestor
  """
  ancestor.children_list = children_list
  ancestor.descendent_list = _create_descendent_list(children_list)
  ancestor.right = children_list[len(children_list)-1]
  ancestor.big_pivot = ancestor.right.big_pivot
  ancestor.left = ancestor.children_list[0]

  # linked list, connecting each child nodes from right to left direction
  for i in reversed(range(1, len(children_list))):
    children_list[i].next = children_list[i-1]


def _update_data(data, data_index: int):
  """
  updates the data list
  index ranges from 0 to number of statistics recorded
  """
  for index, value in data:
    data[data_index][index] += value


def _create_descendent_list(children_list: List[T]) -> List[Sample]:
  """
  replaces children in the children_list with the child's own descendant list
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

def nCr(n, r):
  f = math.factorial
  return f(n) // f(r) // f(n - r)

def nC2(n):
  return nCr(n,2)

def quicksort(children_list: List[T], first: int, last: int):
  """
  modified quicksort that sorts by the value of big_pivot
  """
  if first < last:
    splitpoint = _partition(children_list, first, last)
    quicksort(children_list, first, splitpoint-1)
    quicksort(children_list, splitpoint+1, last)


def _partition(children_list: List[T], first: int, last: int) -> int:
  """
  partitions in place for quicksort
  """
  lo, hi = first + 1, last
  piv = children_list[first].big_pivot
  while True:
    while lo <= hi and children_list[lo].big_pivot <= piv: lo += 1
    while hi >= lo and children_list[hi].big_pivot >= piv: hi -= 1
    if hi < lo: break
    else:
      temp = children_list[lo]
      children_list[lo], children_list[hi] = children_list[hi], temp
  if hi == first: return hi
  part = children_list[first]
  children_list[first], children_list[hi] = children_list[hi], part
  return hi

def display_tree(root: T, time_mode=True, verbose=False):
  """
  displays the tree's Newick representation
  """
  from Bio import Phylo
  # from Bio import /
  from io import StringIO
  newick = traverse(root, time_mode=time_mode)
  tree = Phylo.read(StringIO(str(newick)), 'newick')
  Phylo.draw(tree)
  # if verbose:
  print("\n*** Displaying Each Tree Results ***")
  print(newick)
  print(tree)

def traverse(sample: T, time_mode) -> str:
  """
  iterates through the tree rooted at the sample recursively in pre-order
  builds up a Newick representation
  """
  output = ''
  current = sample.right
  output = _recur_traverse((output + '('), current, time_mode=time_mode)
  while current.next != sample.left:
    current = current.next
    output = _recur_traverse(output + ', ', current, time_mode=time_mode)
  current = sample.left
  output = _recur_traverse(output + ', ', current, time_mode=time_mode) + ')' + str(sample.identity)
  return output


def _recur_traverse(output: str, sample: T, time_mode) -> str:
  """
  appends the sample's information to the current Newick format
  recursively travels to the sample's (right) leaves
  """
  if sample.is_sample():
    # output = output + str(sample.identity) + ':' + str(sample.mutations)
    if time_mode:
      output = output + str(sample.identity) + ':' + str(sample.time)
    else:
      output = output + str(sample.identity) + ':' + str(sample.mutations)
    return output
  current = sample.right
  output = _recur_traverse((output + '('), current, time_mode=time_mode)
  while current.next != sample.left:
    current = current.next
    output = _recur_traverse(output + ', ', current, time_mode=time_mode)
  current = sample.left
  output = _recur_traverse((output + ', '), current, time_mode=time_mode)
  # output = output + ')' + str(sample.identity) + ':' + str(sample.mutations)
  if time_mode:
    output = output + ')' + str(sample.identity) + ':' + str(sample.time)
  else:
    output = output + ')' + str(sample.identity) + ':' + str(sample.mutations)
  return output
