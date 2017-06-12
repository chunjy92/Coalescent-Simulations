import os
import pickle
from functools import partial
from multiprocessing import Pool
from typing import List

from models import MODELS
from simulation import simulate

__author__ = 'Jayeol Chun'

TREE = '.tree'

def load_trees(input_dir:str) -> None:
    """Load Time trees and attach them to respective model objects"""
    assert os.path.exists(input_dir), "Error: Path to the trees does not exist.\nExiting.."
    print("Loading Time Trees..")
    models = [model() for model in MODELS] # init beforehand for classification of tree files
    for file in os.listdir(input_dir):
        if file.endswith(TREE):
            with open(os.path.join(input_dir, file), 'rb') as f:
                filename = file[:file.index('.')]
                model = models[0] if models[0].identity.lower().startswith(filename.lower()) else models[1]
                model.time_trees = pickle.load(f)
    return models

def generate_trees(sample_sizes: List, num_iter: int, num_proc=1, output_dir=None) -> dict:
    print("Generating Time Trees..")
    # out = {}
    models = []
    if num_proc > 1:
        print("Multi-Processing with {} Processes\n".format(num_proc))
        for model in MODELS:
            model = model()
            sim_wrapper = partial(simulate, model=model, num_iter=num_iter)
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
            # out[model.identity] = result
            models.append(model)

    else:
        print("Single-Processing\n")
        for model in MODELS:
            model = model()
            result = {}
            for sample_size in sample_sizes:
                trees = simulate(sample_size, model, num_iter)
                result.update(trees)
            model.time_trees = result
            if output_dir:
                model.save_trees(output_dir)
            # out[model.identity] =result
            models.append(model)
    return models
