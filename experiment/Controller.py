#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import pickle

from .Experiment import Experiment

__author__ = 'Jayeol Chun'


class Controller(object):
  def __init__(self, config):
    # dirs
    self.input = config.input
    self.output = config.output
    self.model_name = config.model_name

    if self.input:
      self.E = self.load()
    else:
      self.E = Experiment(config)

    self.E.display_params()

  def run(self):
    self.E.run()

  def load(self):
    print("[*] Loading from saved Experiment..")
    in_ = os.path.join(self.input, self.model_name)
    try:
      with open(in_, 'rb') as f:
        exp = pickle.load(f)
    except:
      print("Loading Unsuccessful.")
      sys.exit(-1)
    return exp

  def save(self):
    print("[*} Saving Experiment..")
    if not self.output:
      print("[~] Output path not found, Exiting without saving..")
      sys.exit(0)
    out = os.path.join(self.output, self.model_name)
    try:
      with open(out, 'wb') as f:
        pickle.dump(self.E, f, protocol=pickle.HIGHEST_PROTOCOL)
    except:
      print("Saving Unsuccessful.")
      sys.exit(-1)
