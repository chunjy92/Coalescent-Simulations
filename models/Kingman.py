#! /usr/bin/python
# -*- coding: utf-8 -*-
from .Model import Model

import numpy as np

__author__ = 'Jayeol Chun'


class Kingman(Model):
  def __init__(self):
    super().__init__('K', "Kingman")

  def F(self, n, rate=None):
    return n * (n-1) / 2

  def _coalesce(self, coalescent_list):
    time = np.random.exponential(1 / self.F(len(coalescent_list)))
    return time, 2
