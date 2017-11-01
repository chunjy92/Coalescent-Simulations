#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import pickle
from functools import partial
from multiprocessing import Pool

from models import MODELS
from simulation import simulate

__author__ = 'Jayeol Chun'


class Simulation(object):
  def __init__(self, config):
    self.identity = "Simulation Controller"
    self.model_name = "CoalSim.model"

    # directories
    self.output = config.output
    self.input  = config.input

    # sample size params
    self.sample_size      = config.sample_size
    self.sample_size_end  = config.sample_size_end
    self.sample_size_step = config.sample_size_step

    assert self.sample_size_end > self.sample_size, "[!] Sample Size End value should be greater than the init value"
    self.sample_sizes = list(range(self.sample_size, self.sample_size_end, self.sample_size_step))

    # mutation params
    self.mu      = config.mu
    self.mu_step = config.mu_step

    # experiment vars
    self.num_iter = config.num_iter
    self.num_proc = config.num_proc

    # experiment flags
    self.graphics = config.graphics
    self.no_exp   = config.no_exp


  def run(self):
    print("[*] Begin experiment..")
    pass


  def save(self):
    save_path = os.path.join(self.output, self.model_name)
    try:
      print("[*} Saving..")
      with open(save_path, 'wb') as f:
        pickle.dump(self.models, f, protocol=pickle.HIGHEST_PROTOCOL)
    except:
      print("Saving Unsuccessful.")
      sys.exit(-1)


  def load(self):
    print("[*] Loading..")
    load_path = os.path.join(self.input, self.model_name)
    try:
      with open(load_path, 'rb') as f:
        self.models = pickle.load(f)
    except:
      print("Loading Unsuccessful.")
      sys.exit(-1)


  def initModels(self):
    print("[*] Initializing Models..")
    self.models = []
    tree_dict = {}
    if self.num_proc > 1:
      print("Multi-Processing with {} Processes\n".format(self.num_proc))
      for model in MODELS:
        M = model()
        sim_wrapper = partial(simulate, model=M, num_iter=self.num_iter)
        with Pool(processes=self.num_proc) as pool:
          trees = pool.map_async(sim_wrapper, self.sample_sizes)
          pool.close()
          pool.join()
        for tree in trees.get():
          tree_dict.update(tree)
        M.trees = tree_dict
        self.models.append(M)
    else:
      print("Single-Processing\n")
      for model in MODELS:
        M = model()
        for sample_size in self.sample_sizes:
          trees = simulate(sample_size, M, num_iter=self.num_iter)
          tree_dict.update(trees)
        M.trees = tree_dict
        self.models.append(M)


  def setup(self):
    print("[*] Setting up models..")
    if self.input:
      self.load()
    else:
      self.initModels()

