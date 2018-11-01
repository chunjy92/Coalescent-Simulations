#! /usr/bin/python
# -*- coding: utf-8 -*-
import copy
from analyze import Classifier
from models import MODELS, Sample
from utils import collect_nodes
from scipy.stats import poisson
from utils import display_tree

__author__ = 'Jayeol Chun'


class Experiment(object):
  def __init__(self, config):
    print("[*] Initiating Experiment class..")
    self.sample_sizes = range(config.sample_size, config.sample_size_end, config.sample_size_step)
    self.mu = config.mutation_rate
    self.mu_step = config.mutation_rate_step

    self.num_iter = config.num_iter
    self.num_proc = config.num_proc

    self.verbose = config.verbose

    # begin simulation
    self.time_trees = self.simulate()

  def simulate(self):
    print("[*] Begin Simulating..")
    data = {}
    for sample_size in self.sample_sizes:
      for model in MODELS:
        # print("Model:", model)
        trees = {}
        for _ in range(self.num_iter):
          coalescent_list = [Sample(s+1) for s in range(sample_size)]
          root = model.coalesce(coalescent_list, sample_size, verbose=self.verbose)
          if sample_size in trees:
            trees[sample_size].append(root)
          else:
            trees[sample_size] = [root]
        try:
          data[model.identity].update(trees)
        except KeyError:
          data[model.identity] = trees
    print("[*] Done.")
    return data

  def display_params(self):
    print("\nDisplaying params:")
    print("1. Mutation Rate:", self.mu, self.mu_step)
    print("2. Sample Size:", self.sample_sizes)
    print("3. Number of Iterations:", self.num_iter)

  def display_all_trees(self, time_mode=True):
    for treeDict in self.time_trees.values():
      for trees in treeDict.values():
        for tree in trees:
          display_tree(tree, time_mode=time_mode)

  def apply_mutation(self):
    print("[*] Applying Mutations..")
    self.data = {}
    for modelName, treeDict in self.time_trees.items():
      self.data[modelName] = []
      # print(modelName, treeDict)
      for trees in treeDict.values():
        for tree in trees:
          self.data[modelName].append(self._apply_mutation(tree))
    # print(self.data)

  def _apply_mutation(self, tree):
    nodes = collect_nodes(copy.deepcopy(tree))
    # nodes = collect_nodes(tree)
    # display_tree(nodes[0])
    bbl = 0
    sackin_1 = 0
    sackin_2 = 0
    cherry_1 = 0
    cherry_2 = 0

    for node in nodes:
      if not node.is_sample() and node.descendent_list == node.children_list:
        cherry_1 += 1

    for node in reversed(nodes):
      node.mutations = poisson.rvs(self.mu * node.time)
      # print(node)
      # print ("Node depth:", node.depth)
      # print("Node Mutation:", node.mutations)
      if not node.is_sample():
        # print(node.descendent_list == node.children_list)
        for child in node.children_list:
          if not child.is_sample() and child.mutations==0:
            idx = node.children_list.index(child)

            # delete ancestor with mutation value of 0
            del node.children_list[idx]

            # insert its children list in the deleted ancestor's position
            node.children_list[idx:idx] = child.children_list

            # reconfigure
            node.right = node.children_list[len(node.children_list)-1]
            node.big_pivot = node.right.big_pivot
            node.left = node.children_list[0]

            # re-join children linked lists
            for i in reversed(range(1, len(node.children_list))):
              node.children_list[i].next = node.children_list[i-1]

            for desc in node.descendent_list:
              # print("Decreasing depth for Desc List")
              # print(desc)
              # print(desc.depth)
              desc.depth -= 1
              # print(desc.depth)



      else: # is a leaf node
        bbl += node.mutations
        # if node.mutations == 0:
        #   node.depth -= 1
        # if node.depth == 0:
        #   continue
        sackin_1 += node.depth - 1
    # display_tree(nodes[0], time_mode=False)


    # print("Second iter")
    for node in reversed(nodes):
      if node.is_sample():
        if node.depth == 0:
          continue
        # print(node)
        # print("Node depth:", node.depth)
        # print("Node Mutation:", node.mutations)
        sackin_2 += node.depth - 1
      else:
        if node.children_list == node.descendent_list and node.mutations > 0:
          # print("haha")
          # print(node)
          cherry_2 += 1

    # print("Sackin 1:", sackin_1)
    # print("Sackin 2:", sackin_2)
    #
    # print("cherry 1:", cherry_1)
    # print("cherry 2:", cherry_2)

    # number of cherries as a feature, count


    return [bbl, cherry_2, sackin_2]

  def analyze(self):
    self.C = Classifier(self.data)
    pass

  def run(self):
    self.apply_mutation()
    self.analyze()
