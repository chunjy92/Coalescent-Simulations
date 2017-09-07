# -*- coding: utf-8 -*-

from typing import TypeVar

__author__ = 'Jayeol Chun'


class Sample:  # Leaf
  def __init__(self, identity_count: int):
    self.identity = str(identity_count)
    self.big_pivot = identity_count     # used as key values in quicksort: largest identity # of its subtree
    self.next = None                    # links to its left neighbor child of its parent
    self.time = 0                       # time to the previous coalescent event
    self.generation = 0                 # each coalescent event represents one generation
    self.pairwise_dist = 0.             # contibution to overall average pairwise distance
    self.mutations = 0

  def __repr__(self):
    # return 'Sample {} with Mutations {}.'.format(self.identity, self.mutations)
    return 'Sample {} with Time branch value {:.7f}'.format(self.identity, self.time)

  def is_sample(self):
    return all(word not in self.identity for word in ['A', 'K', 'B'])

class Ancestor(Sample):  # Internal Node
  def __init__(self, identity_count: int):
    super().__init__(identity_count)
    self.identity = 'A{}'.format(identity_count)
    self.generation = identity_count
    self.left = None           # left-most child
    self.right = None          # right-most child
    self.descendent_list = []  # all samples below it
    self.children_list = []    # all children directly below it

  def __repr__(self):
    return super().__repr__().replace('Sample', 'Ancestor')


T = TypeVar('T', Sample, Ancestor)
