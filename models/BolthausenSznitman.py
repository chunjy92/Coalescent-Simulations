#! /usr/bin/python
# -*- coding: utf-8 -*-
from .Model import Model

import numpy as np

__author__ = 'Jayeol Chun'


class BolthausenSznitman(Model):
  def __init__(self):
    super().__init__('B', "Bolthausen Sznitman")

  def F(self, n, rate):
    total_rate = 0
    for i in range(n-1):
      m = i + 2
      i_rate = n / (m * (m-1))
      rate[i] = i_rate
      total_rate += i_rate
    return rate, total_rate

  def _coalesce(self, coalescent_list):
    l = len(coalescent_list)
    m_list = np.arange(2, l+1)
    bsf_rate = np.zeros(l-1)
    mn_rate, total_rate = self.F(l, np.zeros(l-1))
    for j in range(0, np.size(mn_rate)):
      bsf_rate[j] = mn_rate[j] / total_rate
    num_children = np.random.choice(m_list, 1, replace=False, p=bsf_rate)
    time = np.random.exponential(1 / total_rate)
    return time, num_children
