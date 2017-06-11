# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import poisson
from typing import List
from .structure import Sample, Ancestor, T

__author__ = 'Jayeol Chun'


def update_children(ancestor: Ancestor, children: List[T],
                    gen_time: np.ndarray, verbose: bool = False):
    '''
    for each child node under ancestor, calculate time and mutation value
    '''
    # BEGIN: iteration through the children_list
    for child in children:
        _update_time(child, ancestor, gen_time)
        # child.mutations = poisson.rvs(self.mu * child.time)

    # Update new information to the ancestor
    _update_ancestor(ancestor, children)


def _update_time(sample: T, ancestor: Ancestor, gen_time: np.ndarray):
    """
    adds up the between-generation time
    """
    for j in range(ancestor.generation-1, sample.generation-1, -1):
        sample.time += gen_time[j]


def _update_ancestor(ancestor: Ancestor, children_list: List[T]):
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

def display_tree(root: T, verbose=False):
    """
    displays the tree's Newick representation
    """
    from Bio import Phylo
    # from Bio import /
    from io import StringIO
    newick = _traversal(root)
    tree = Phylo.read(StringIO(str(newick)), 'newick')
    Phylo.draw(tree)
    if verbose:
        print("\n*** Displaying Each Tree Results ***")
        print(newick)
        print(tree)

def _traversal(sample: T) -> str:
    """
    iterates through the tree rooted at the sample recursively in pre-order
    builds up a Newick representation
    """
    output = ''
    current = sample.right
    output = _recur_traversal((output + '('), current)
    while current.next != sample.left:
        current = current.next
        output = _recur_traversal(output + ', ', current)
    current = sample.left
    output = _recur_traversal(output + ', ', current) + ')' + str(sample.identity)
    return output


def _recur_traversal(output: str, sample: T) -> str:
    """
    appends the sample's information to the current Newick format
    recursively travels to the sample's (right) leaves
    """
    if sample.is_sample():
        # output = output + str(sample.identity) + ':' + str(sample.mutations)
        output = output + str(sample.identity) + ':' + str(sample.time)
        return output
    current = sample.right
    output = _recur_traversal((output + '('), current)
    while current.next != sample.left:
        current = current.next
        output = _recur_traversal(output + ', ', current)
    current = sample.left
    output = _recur_traversal((output + ', '), current)
    # output = output + ')' + str(sample.identity) + ':' + str(sample.mutations)
    output = output + ')' + str(sample.identity) + ':' + str(sample.time)
    return output