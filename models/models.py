# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple
from .model_template import CoalescentModel
from .structure import *

__author__ = 'Jayeol Chun'


class Kingman(CoalescentModel):
    def __init__(self):
        super().__init__()
        self.identity = 'K'

    def _F(self, n: int, rate=None) -> float:
        '''
        Kingman Function
        '''
        return n * (n-1) / 2

    def _coalesce_aux(self, coalescent_list):
        time = np.random.exponential(1 / self._F(len(coalescent_list)))
        return time, 2


class BolthausenSznitman(CoalescentModel):
    def __init__(self):
        super().__init__()
        self.identity = 'B'

    def _F(self, n: int, rate: np.ndarray=None) -> Tuple[np.ndarray, float]:
        """
        Bolthausen-Sznitman function
        """
        total_rate = 0
        for i in range(n - 1):
            m = i + 2
            i_rate = n / (m * (m - 1))
            rate[i] = i_rate
            total_rate += i_rate
        return rate, total_rate

    def _coalesce_aux(self, coalescent_list):
        l = len(coalescent_list)
        m_list = np.arange(2, l+1)
        # mn_rate = np.zeros(len(coalescent_list) - 1)
        bsf_rate = np.zeros(l-1)
        mn_rate, total_rate = self._F(l, np.zeros(l-1))
        for j in range(0, np.size(mn_rate)):
            bsf_rate[j] = mn_rate[j] / total_rate
        num_children = np.random.choice(m_list, 1, replace=False, p=bsf_rate)
        time = np.random.exponential(1/total_rate)
        return time, num_children


M = TypeVar('M', Kingman, BolthausenSznitman)
MODELS = [Kingman, BolthausenSznitman]
