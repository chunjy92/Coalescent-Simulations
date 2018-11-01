#! /usr/bin/python
# -*- coding: utf-8 -*-
import math

__author__ = 'Jayeol Chun'


def quicksort(children_list, first, last):
  """
  modified quicksort that sorts by the value of big_pivot
  """
  if first < last:
    splitpoint = _partition(children_list, first, last)
    quicksort(children_list, first, splitpoint-1)
    quicksort(children_list, splitpoint+1, last)

def _partition(children_list, first, last):
  """
  partitions in place for quicksort
  """
  lo, hi = first+1, last
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
