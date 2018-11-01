#! /usr/bin/python
# -*- coding: utf-8 -*-
import math
import numpy as np

__author__ = 'Jayeol Chun'


def nCr(n, r):
  f = math.factorial
  return f(n) // f(r) // f(n - r)

def nC2(n):
  return nCr(n,2)

def project_onto_plane(a, b):
  """
  finds the vector projection of points onto the hyperplane
  @param a : Tuple - coefficients of the hyperplane
  @param b : Tuple - original vector
  @return  : Tuple - new vector projected onto the hyperplane
  """
  dot = np.dot(a, b) / np.linalg.norm(a)
  p = dot * a / np.linalg.norm(a)
  return b - p