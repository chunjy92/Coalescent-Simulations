#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Jayeol Chun'


class Sample:
  """Leaf node"""
  def __init__(self, identity_count):
    self.identity = str(identity_count)
    self.big_pivot = identity_count     # largest identity # of descendant, used in quicksort
    self.next = None                    # links to its left sibling
    self.time = 0                       # time to the previous coalescent event
    self.generation = 0                 # each coalescent event represents one generation
    self.pairwise_dist = 0.             # contribution to overall average pairwise distance

  def __repr__(self):
    # return 'Sample {} with time to coalescent event: {:.4f}'.format(self.identity, self.time)
    return 'Sample {} with time to coalescent event: {:.4f}'.format(self.identity, self.time)
    # return 'Sample {}'.format(self.identity)

  def is_sample(self):
    return all(word not in self.identity for word in ['A', 'K', 'B'])


class Ancestor(Sample):
  """Internal Node"""
  def __init__(self, identity_count):
    super().__init__(identity_count)
    self.identity = 'A{}'.format(identity_count)
    self.generation = identity_count
    self.left = None           # left-most child
    self.right = None          # right-most child
    self.descendent_list = []  # all samples below it
    self.children_list = []    # all children immediately below it

  def __repr__(self):
    return super().__repr__().replace('Sample', 'Ancestor')
