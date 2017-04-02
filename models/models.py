# -*- coding: utf-8 -*-                     #
# ========================================= #
# Models                                    #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 04/02/2017                  #
# ========================================= #

import numpy as np
from typing import TypeVar, List, Tuple
from models.structure import *
from models.model_template import Model

__author__ = 'Jayeol Chun'

T = TypeVar('T', Sample, Ancestor)

class Kingman(Model):
    def __init__(self, n: int, mu: float, coalescent_list: List[T] = None,
                 data: List[float] = None):  # change data typing
        super().__init__(n, mu)
        self.coalescent_list = coalescent_list
        self.data = data

    def F(self, n: int, rate=None) -> float:
        '''
        Kingman Function
        '''
        return n * (n - 1) / 2

    def coalesce(self, coalescent_list: List[Sample],
                 data: Tuple[int, np.ndarray], verbose=False) -> Ancestor:
        '''

        :param coalescent_list:
        :param data:
        :param verbose:
        :return:
        '''
        self.coalescent_list = coalescent_list  # for this particular run
        self.data = data[1]
        data_idx = data[0]

        generation_time = np.zeros(self.n-1)  # coalescent time for each generation, fixed: numpy array
        nth_coalescence = 0  # Starting from 0

        # until only the most recent common ancestor remains
        while len(coalescent_list) > 1:
            # calculate time
            generation_time[nth_coalescence] = np.random.exponential(1 / self.F(len(coalescent_list)))
            nth_coalescence += 1

            # merge 2 children
            merged_ancestor, children = self.merge(nth_coalescence, 2)

            # update these children
            self.update_children(merged_ancestor, children, generation_time, data_idx)

            if verbose:
                print("\n*** Merging happened!")
                print("----> {} was created after merging "
                      "{} and {}".format(merged_ancestor.identity, children[0].identity, children[1].identity))
                print("Now, the coalescent list is:")
                print(coalescent_list)

        assert len(coalescent_list) == 1 and type(coalescent_list[0]) is Ancestor, \
            "There was an error in simulating a Coalescent event." \
            "Make sure you are providing at least two Samples for the merging to take place."
        root = coalescent_list.pop()
        root.identity = root.identity.replace('A','K')
        if verbose:
            print("Coalescing Complete.")
            print(generation_time)

        return root


class BolthausenSznitman(Model):
    def __init__(self, n: int, mu: float, coalescent_list: List[T]=None,
                 data: List[float]=None):  # change data typing
        super().__init__(n, mu)
        self.coalescent_list = coalescent_list
        self.data = data

    def F(self, n: int, rate: np.ndarray=None) -> Tuple[np.ndarray, float]:
        """
        Bolthausen-Sznitman function
        @param mn_rate : 1-d Array -
        @param n       : Int       - current size of the coalescent list, i.e. number of samples available for coalescence
        @return        : Tuple     - ( mn_rate    : 1-d Array -
                                       total_rate : 1-d Array - holds each i rate )
        """
        total_rate = 0
        for i in range(n - 1):
            m = i + 2
            i_rate = n / (m * (m - 1))
            rate[i] = i_rate
            total_rate += i_rate
        return rate, total_rate

    def coalesce(self, coalescent_list: List[Sample],
                 data: Tuple[int, np.ndarray], verbose=False) -> Ancestor:
        """
        models the Bolthausen-Sznitman coalescence
        @param sample_size     : Int       - refer to argument of src
        @param mu              : Float     - refer to argument of src
        @param coalescent_list : 1-d Array - holds samples
        @param data            : Tuple     - (data_list, data_index) -> refer to __update_data
        @param newick          : Bool      - refer to argument of src
        """
        self.coalescent_list = coalescent_list  # for this particular run
        self.data = data[1]
        data_idx = data[0]

        generation_time = np.zeros(self.n - 1)  # coalescent time for each generation, fixed: numpy array
        nth_coalescence = 0  # Starting from 0

        # until only the most recent common ancestor remains
        while len(coalescent_list) > 1:
            # Time and Number of Children Calculation
            m_list = np.arange(2, len(coalescent_list) + 1)
            # mn_rate = np.zeros(len(coalescent_list) - 1)
            bsf_rate = np.zeros(len(coalescent_list) - 1)
            mn_rate, total_rate = self.F(len(coalescent_list), np.zeros(len(coalescent_list) - 1))
            for j in range(0, np.size(mn_rate)):
                bsf_rate[j] = mn_rate[j] / total_rate
            num_children = np.random.choice(m_list, 1, replace=False, p=bsf_rate)
            generation_time[nth_coalescence] = np.random.exponential(1 / total_rate)
            nth_coalescence += 1

            # merge children
            merged_ancestor, children = self.merge(nth_coalescence, num_children)

            self.update_children(merged_ancestor, children, generation_time, data_idx)

            if verbose:
                print("\n*** Merging happened!")
                print("----> {} was created after merging ".format(merged_ancestor.identity) + ', '.join("{}".format(child.identity) for child in children))
                print("Now, the coalescent list is:")
                print(coalescent_list)

        assert len(coalescent_list) == 1 and type(coalescent_list[0]) is Ancestor, \
            "There was an error in simulating a Coalescent event." \
            "Make sure you are providing at least two Samples for the merging to take place."
        root = coalescent_list.pop()
        root.identity = root.identity.replace('A', 'B')
        if verbose:
            print("Coalescing Complete.")
            print(generation_time)
        return root