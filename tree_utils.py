import os
import sys
import pickle
from random import shuffle
from functools import partial
from multiprocessing import Pool
from typing import List

from scipy.special import comb

from models import MODELS
from simulation import simulate

__author__ = 'Jayeol Chun'

TREE = '.tree'

def load_trees(input_dir:str) -> None:
    """Load Time trees and attach them to respective model objects"""
    assert os.path.exists(input_dir), "Error: Path to the trees does not exist.\nExiting.."
    print("******* Loading Time Trees *******")
    # ..init beforehand for classification of tree files
    models = [model() for model in MODELS]
    for file in os.listdir(input_dir):
        if file.endswith(TREE):
            with open(os.path.join(input_dir, file), 'rb') as f:
                filename = file[:file.index('.')]
                model = models[0] if models[0].identity.lower().startswith(filename.lower()) else models[1]
                model.time_trees = pickle.load(f)
                print(model.identity)
                print(model.time_trees)


def generate_trees(sample_sizes: List, num_iter: int, num_proc=1, output_dir=None) -> dict:
    print("******* Generating Time Trees *******")
    out = {}
    if num_proc > 1:
        print("******* Multi-Processing with {} Processes *******".format(num_proc))
        for model in MODELS:
            model = model()
            sim_wrapper = partial(simulate, model=model, num_iter=num_iter)
            # shuffle(sample_sizes)
            result = {}
            with Pool(processes=num_proc) as pool:
                res = pool.map_async(sim_wrapper, sample_sizes)
                pool.close()
                pool.join()
            for tree in res.get():
                result.update(tree)

            model.time_trees = result
            if output_dir:
                model.save_trees(output_dir)

            out[model.identity] = result

    else:
        print("******* Single-Processing *******")
        for model in MODELS:
            model = model()
            result = {}
            for sample_size in sample_sizes:
                trees = simulate(sample_size, model, num_iter)
            # model.time_trees.update(trees)
                result.update(trees)
            model.time_trees = result
            if output_dir:
                model.save_trees(output_dir)
            out[model.identity] =result
    return out

def heterozygosity_calculator( sample_size, k):
    '''
    helper method to calculate mean separation time
    '''
    b = comb(sample_size, 2)
    a = k * (sample_size-k)
    # return a/b * sample.mutations
    return a/b
