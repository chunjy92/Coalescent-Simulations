# -*- coding: utf-8 -*-                     #
# ========================================= #
# Model Template                            #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 03/27/2017                  #
# ========================================= #

import numpy as np
from scipy.stats import poisson
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, TypeVar

from utils.sorting import quicksort
from models.update import *
from models.structure import *

__author__ = 'Jayeol Chun'

T = TypeVar('T', Sample, Ancestor)

class Model(ABC):
    def __init__(self, n, mu):
        '''
        initialize final constants
        :param n: init size of population, i.e. number of init samples or leaves
        :param mu: mutation rate
        '''
        self.n = n
        self.mu = mu
        self._data = []
        self._coalescent_list = []

    @property
    def coalescent_list(self):
        return self._coalescent_list

    @coalescent_list.setter
    def coalescent_list(self, new_list: List[T]=None):
        self._coalescent_list = new_list

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_list: List[T]=None):
        self._data = new_list

    def export_data(self):
        pass

    @abstractmethod
    def F(self, n: int, rate: np.ndarray=None):
        pass

    @abstractmethod
    def coalesce(self, coalescent_list: List[Sample],
                 data: np.ndarray, verbose=False) -> Ancestor:
        pass


    def merge(self, merge_identity, num_children=2):
        '''
        Perform a coalescent event that creates a merged ancestor
        '''

        # Create an Internal Node Representing a Coalescent Event
        merged_ancestor = Ancestor(merge_identity)

        # The merge_sample's immediate children chosen
        children = np.random.choice(self.coalescent_list, num_children, replace=False).tolist()

        # sort by big pivot for visual ease
        quicksort(children, 0, len(children)-1)
        self.coalescent_list.append(merged_ancestor)
        return merged_ancestor, children


    def update_children(self, ancestor, children, gen_time, data_index, verbose=False):
        '''
        For each child node under ancestor, calculate time and mutation value
        Then, update accordingly
        '''

        temp_list = children[:]
        children_index = 0  # index of changing children_list

        ##########################################################################################
        # BEGIN: iteration through the children_list
        while children_index < np.size(temp_list):
            current = temp_list[children_index]  # current child under inspection

            update_time(current, ancestor, gen_time)
            current.mutations = poisson.rvs(self.mu*current.time)

            # First Case : a Sample (Leaf Node)
            if current.is_sample():
                update_data(self.data, data_index, *zip((0,), (current.mutations,)))

            # Second Case : an Internal Node with Mutations == 0
            elif current.mutations==0:
                # Delete this Current Child from the Coalescent List
                del self.coalescent_list[self.coalescent_list.index(current)]

                # Replace this Current Child with its children nodes
                del temp_list[children_index]
                temp_list[children_index:children_index] = current.children_list

                # Create Linked List that Connects the Replacing children_list with the original children
                # on its left if it exists
                if children_index > 0:
                    temp_list[children_index].next = temp_list[children_index-1]
                # Increase the index appropriately by jumping over the current child's children
                children_index += len(current.children_list)-1

            # Delete Current Child from the Coalescent List (unless Deleted alrdy in the Second Case)
            current = temp_list[children_index]
            if current in self.coalescent_list:
                del self.coalescent_list[self.coalescent_list.index(current)]

            # Link current child to its left
            if children_index > 0:
                current.next = temp_list[children_index-1]

            # increase indices
            children_index += 1
        # END: iteration through the children_list
        ##########################################################################################

        # Update new information to the ancestor
        update_ancestor(ancestor, temp_list)