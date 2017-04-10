# -*- coding: utf-8 -*-

from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import poisson
from .structure import *
from .utils import quicksort, update_time, update_ancestor

__author__ = 'Jayeol Chun'


class Model(ABC):
    def __init__(self, n: int, mu: float):
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
    def coalesce(self, coalescent_list: List[Sample],data: np.ndarray,
                 exp=False, verbose=False) -> Ancestor:
        pass

    def update_data(self, data_index: int, *data: Tuple[int, float]):
        """
        updates the data list
        index ranges from 0 to number of statistics recorded
        """
        for index, value in data:
            self.data[data_index][index] += value


    def merge(self, merge_identity: int, num_children: int=2) -> Tuple[Ancestor, List[T]]:
        '''
        performs a coalescent event that creates a merged ancestor
        -> returns the new ancestor and its children list
        '''

        # Create an Internal Node Representing a Coalescent Event
        merged_ancestor = Ancestor(merge_identity)

        # The merge_sample's immediate children chosen
        children = np.random.choice(self.coalescent_list, num_children, replace=False).tolist()

        # sort by big pivot for visual ease
        quicksort(children, 0, len(children)-1)
        self.coalescent_list.append(merged_ancestor)
        return merged_ancestor, children


    def update_children(self, ancestor: Ancestor, children: List[T],
                        gen_time: np.ndarray, data_index: int, verbose: bool=False):
        '''
        for each child node under ancestor, calculate time and mutation value
        -> then, update accordingly
        '''
        temp_list = children[:]
        children_index = 0  # index of changing children_list

        # BEGIN: iteration through the children_list
        while children_index < np.size(temp_list):
            current = temp_list[children_index]  # current child under inspection

            update_time(current, ancestor, gen_time)
            current.mutations = poisson.rvs(self.mu*current.time)

            # First Case : a Sample (Leaf Node)
            if current.is_sample():
                self.update_data(data_index, (0, current.mutations)) # 0, when only considering bottom branch length

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

            children_index += 1

        # Update new information to the ancestor
        update_ancestor(ancestor, temp_list)
