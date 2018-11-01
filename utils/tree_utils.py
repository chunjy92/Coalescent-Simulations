#! /usr/bin/python
# -*- coding: utf-8 -*-
from io import StringIO
from Bio import Phylo

__author__ = 'Jayeol Chun'


def display_tree(root, time_mode=True):
  """
  displays the tree's Newick representation
  """
  newick = traverse(root, time_mode=time_mode)
  tree = Phylo.read(StringIO(str(newick)), 'newick')
  Phylo.draw(tree)

def traverse(sample, time_mode):
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

def _recur_traverse(output, sample, time_mode):
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


def collect_nodes(sample):
  nodes = [sample]
  depth = 0
  current = sample.right
  _recurse(current, nodes, depth+1)
  while current.next != sample.left:
    current = current.next
    _recurse(current, nodes, depth+1)
  current = sample.left
  _recurse(current, nodes, depth+1)
  sample.depth = 0
  return nodes


def _recurse(sample, nodes, depth):
  nodes.append(sample)
  sample.depth = depth
  if sample.is_sample():
    return
  current = sample.right
  _recurse(current, nodes, depth+1)
  while current.next != sample.left:
    current = current.next
    _recurse(current, nodes, depth+1)
  current = sample.left
  _recurse(current, nodes, depth+1)
