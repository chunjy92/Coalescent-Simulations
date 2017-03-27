# -*- coding: utf-8 -*-                     #
# ========================================= #
# Models                                    #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 03/26/2017                  #
# ========================================= #

import numpy as np
from scipy.stats import poisson
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, TypeVar

from models.update import *
from models.structure import Sample, Ancestors
from utils.sorting import quicksort
from utils.utils import check_all


T = TypeVar('T', Sample, Ancestors)

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
    def data(self, new_list: List[T] = None):
        self._data = new_list

    def export_data(self):
        pass

    @abstractmethod
    def F(self, n: int, rate: List[int]=None):
        pass

    @abstractmethod
    def coalesce(self, coalescent_list, data):
        pass


    def merge(self, merge_identity, num_children=2):
        """
        Given a number of children to be merged, perform a coalescent event that creates a merged ancestor
        @param coalescent_list : 1-d Array - holds samples currently available for new coalescence
        @param identity_count  : Int       - distinct ID number to create a new merged sample
        @param num_children    : Int       - number of children to be coalesced, 2 by default
        @return                : Tuple     - ( coalescent_list : 1-d Array - updated coalescent list
                                               merge_sample    : Ancestor  - new merged sample
                                               children_list   : 1-d Array - Ancestor's direct children )
        """
        # Create an Internal Node Representing a Coalescent Event
        merged_ancestor = Ancestors(merge_identity)

        # The merge_sample's immediate children chosen
        children = np.random.choice(self.coalescent_list, num_children, replace=False)
        quicksort(children, 0, np.size(children)-1)  # sort by big pivot for visual ease
        # self.coalescent_list.append(merged_ancestor)
        self.coalescent_list = np.append(self.coalescent_list, merged_ancestor)
        return merged_ancestor, children



    # def __update_children(self, ancestor, data_list, data_index, coalescent_list, children_list, gen_time):
    def update_children(self, ancestor, children, gen_time, data_index):
        """
        A.for each child node under the ancestor, do:
            1) calculate its time, taking into account the generation difference between the sample and its ancestor
            2) based on 1), calculate the branch's mutation value
            3) perform appropriate tasks depending on what type the child is -> refer to comments below for details
        B. update the ancestor
        @param mu               : Float     - refer to argument of src
        @param ancestor         : Ancestor  - newly merged ancestor
        @param data_list        : 2-d Array - to be updated
        @param data_index       : Int       - ensures each data is stored at right place
        @param coalescent_list  : 1-d Array - initial coalescent list
        @param children_list    : 1-d Array - for analysis of each child in the list
        @param gen_time         : 1-d Array - used to update time of each child to the ancestor
        @return coalescent_list : 1-d Array - updated coalescent list
        """
        temp_list = np.copy(children)
        children_index = 0  # index of changing children_list

        ##########################################################################################
        # BEGIN: iteration through the children_list

        while children_index < np.size(temp_list):
            current = temp_list[children_index]  # current child under inspection

            update_time(current, ancestor, gen_time)
            current.mutations = poisson.rvs(self.mu * current.time)

            # First Case : a Sample (Leaf Node)
            if current.is_sample():
                update_data(self.data, data_index, *zip((0,), (current.mutations,)))

            # Second Case : an Internal Node with Mutations == 0
            elif not current.is_sample() and current.mutations==0:
                # Delete this Current Child from the Coalescent List
                # print(self.coalescent_list)
                # print(temp_list)
                # print(self.coalescent_list == temp_list[children_index])
                self.coalescent_list = np.delete(self.coalescent_list, int(np.where(
                    self.coalescent_list == temp_list[children_index] )[0]))

                # Replace this Current Child with its children nodes
                temp_list = np.insert(temp_list, children_index, current.children_list)
                temp_list = np.delete(temp_list, children_index + np.size(current.children_list))

                # Create Linked List that Connects the Replacing children_list with the original children on its left if it exists
                if children_index > 0:
                    temp_list[children_index].next = temp_list[children_index - 1]
                # Increase the index appropriately by jumping over the current child's children
                children_index += (np.size(current.children_list) - 1)

            # Third Case : an Internal Node with Mutations > 0
            # else:
            #    __update_data(data_list, data_index, *zip((0,), (current.mutations,)))

            # Delete Current Child from the Coalescent List (unless Deleted alrdy in the Second Case)
            cond = self.coalescent_list == temp_list[children_index]
            if True in cond:
                self.coalescent_list = np.delete(self.coalescent_list, int(np.where(cond)[0]))

            # Link current child to its left
            if children_index > 0:
                current.next = temp_list[children_index - 1]

            # increase indices
            children_index += 1
        # END: iteration through the children_list
        ##########################################################################################


        # Update new information to the ancestor
        update_ancestor(ancestor, temp_list)

        return self.coalescent_list


class Kingman(Model):
    def __init__(self, n: int, mu: int, coalescent_list: List[T]=None, data: List[float]=None): # change data typing
        super().__init__(n, mu)
        self.coalescent_list = coalescent_list
        self.data = data

    def F(self, n: int, rate: List[int]=None) -> int:
        '''
        Kingman Function
        :param n: Size of current population (size of coalescent list at that moment in time)
        :return:  Number of Possible pairs
        '''
        return n*(n-1)/2


    def coalesce(self, coalescent_list: List[int],
                 data: Tuple[int, List[List[float]]]):
        self.coalescent_list = coalescent_list # set for this particular run
        data_idx = data[0]
        self.data = data[1]
        generation_time = np.zeros(self.n - 1)  # coalescent time for each generation
        nth_coalescence = 0  # Starting from 0

        # until only the most recent common ancestor left
        while np.size(coalescent_list) > 1:
            # calculate time
            generation_time[nth_coalescence] = np.random.exponential(1 / self.F(np.size(self.coalescent_list)))
            nth_coalescence += 1

            # merge 2 children
            merged_ancestor, children = self.merge(nth_coalescence)

            # update these children
            self.coalescent_list = self.update_children(merged_ancestor, children, generation_time, data_idx)

            if check_all(self.coalescent_list, Ancestors):
                break

        # print(self.coalescent_list[0])
        # display_tree((self.coalescent_list[0], ))
        return self.coalescent_list[0]




class BolthausenSznitman(Model):
    def __init__(self, n):
        self.n = n
        pass

    def __b_F(self, mn_rate, n):
        """
        computes the Bolthausen-Sznitman function
        @param mn_rate : 1-d Array -
        @param n       : Int       - current size of the coalescent list, i.e. number of samples available for coalescence
        @return        : Tuple     - ( mn_rate    : 1-d Array -
                                       total_rate : 1-d Array - holds each i rate )
        """
        total_rate = 0
        for i in range(n-1):
            m = i + 2
            i_rate = n / (m * (m-1))
            mn_rate[i] = i_rate
            total_rate += i_rate
        return mn_rate, total_rate

    def __bs_coalescence(self, sample_size, mu, coalescent_list,
                         *data, newick=False):
        """
        models the Bolthausen-Sznitman coalescence
        @param sample_size     : Int       - refer to argument of src
        @param mu              : Float     - refer to argument of src
        @param coalescent_list : 1-d Array - holds samples
        @param data            : Tuple     - (data_list, data_index) -> refer to __update_data
        @param newick          : Bool      - refer to argument of src
        """
        gen_time = np.zeros(sample_size - 1)  # coalescent time for each generation
        nth_coalescence = 0  # Starting from 0

        # Until reaching the Most Recent Common Ancestor
        while np.size(coalescent_list) > 1:
            # Time and Number of Children Calculation
            m_list = np.arange(2, np.size(coalescent_list) + 1)
            mn_rate = np.zeros(np.size(coalescent_list) - 1)
            bsf_rate = np.zeros(np.size(coalescent_list) - 1)
            mn_rate, total_rate = __b_F(mn_rate, np.size(coalescent_list))
            for j in range(0, np.size(mn_rate)):
                bsf_rate[j] = mn_rate[j] / total_rate
            num_children = np.random.choice(m_list, 1, replace=False, p=bsf_rate)
            time = np.random.exponential(1 / total_rate)
            gen_time[nth_coalescence] = time
            nth_coalescence += 1

            # merged ancestor of the coalescent event and its children obtained
            consts = {'identity_count': nth_coalescence, 'num_children': num_children}
            coalescent_list, ancestor, children_list = __coalesce_children(coalescent_list, **consts)

            # update the tree using mutations as branch length, recording data along the way
            lists = {'coalescent_list': coalescent_list, 'children_list': children_list, 'gen_time': gen_time}

            coalescent_list = __update_children(mu, ancestor, *data, **lists)
            if all_ancestors(coalescent_list):
                break
