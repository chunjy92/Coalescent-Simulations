# -*- coding: utf-8 -*-                     #
# ========================================= #
# Models Update Utils                       #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 03/27/2017                  #
# ========================================= #

from typing import TypeVar, List
from models.structure import *

__author__ = 'Jayeol Chun'

T = TypeVar('T', Sample, Ancestor)

def update_time(sample: Ancestor, ancestor: Ancestor, gen_time: List[float]):
    """
    adds up the between-generation time
    @param sample   : Ancestor / Sample - sample whose coalescent time to its ancestor is to be calculated
    @param ancestor : Ancestor          - newly merged ancestor
    @param gen_time : 1-d Array         - holds coalescent time between generations
    """
    for j in range(ancestor.generation-1, sample.generation-1, -1):
        sample.time += gen_time[j]


# update needed
def update_data(data_list, data_index, *data):
    """
    updates the data list
    @param data_list  : 2-d Array - holds overall data
    @param data_index : Int       - ensures each data is stored at right place
    @param data       : Tuple     - (index, value) where the value is to be added to the data_list at the index
    """
    for index, value in data:
        data_list[data_index][index] += value


def update_ancestor(ancestor: Ancestor, children_list: List[T]):
    """
    assigns new attributes to the merged ancestor
    @param ancestor      : Ancestor  - newly merged ancestor, represents a single coalescent event
    @param children_list : 1-d Array - nodes that are derived from the ancestor
    """
    ancestor.children_list = children_list
    ancestor.descendent_list = _create_descendent_list(children_list)
    ancestor.right = children_list[len(children_list)-1]
    ancestor.big_pivot = ancestor.right.big_pivot
    ancestor.left = ancestor.children_list[0]


def _create_descendent_list(children_list: List[T]) -> List[Sample]:
    """
    creates a descendent list by replacing samples in the children list with its own descendent list
    @param children_list    : 1-d Array - for each children in the list, see what samples are below it and compile them
    @return descendent_list : 1-d Array - newly created descendent_list
    """
    descendent_list = children_list[:]
    i = 0
    while i < len(descendent_list):
        if descendent_list[i].is_sample():
            i += 1
        else:
            # insert the internal node's own descendent list at the node's index in the current descendent_list
            # -> since the node is below the sample, its descdent list must have already been updated
            size = len(descendent_list[i].descendent_list)

            descendent_list[i:i] = descendent_list[i].descendent_list
            del descendent_list[i+size]

            i += size
    return descendent_list