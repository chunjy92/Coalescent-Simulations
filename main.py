#! /usr/bin/python
# -*- coding: utf-8 -*-
import time

from config import get_config
from experiment import Controller

__author__ = 'Jayeol Chun'


def main(config):
  tic = time.time()
  print("******* Coalescent Simulations Main *******")

  # init Experiment class
  controller = Controller(config)
  controller.run()
  controller.save()

  print("\n******* Program Execution Time: {:.2f} s *******".format(time.time()-tic))

if __name__=='__main__':
  main(get_config())
