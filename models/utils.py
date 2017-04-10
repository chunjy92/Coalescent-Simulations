# -*- coding: utf-8 -*-

import numpy as np
from typing import List
from .structure import *

__author__ = 'Jayeol Chun'


def update_time(sample: Ancestor, ancestor: Ancestor, gen_time: np.ndarray):
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
