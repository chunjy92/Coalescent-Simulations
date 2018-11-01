#! /usr/bin/python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser

__author__ = 'Jayeol Chun'


parser = ArgumentParser()

# sample size related params
parser.add_argument("--sample_size", type=int, default=30,
                    help="init population sample size")
parser.add_argument("--sample_size_end", type=int, default=40,
                    help="end of sample size range")
parser.add_argument("--sample_size_step", type=int, default=5,
                    help="sample size range step unit")
# mutation rate related params
parser.add_argument("--mutation_rate", type=float, default=.8,
                    help="init mutation rate")
parser.add_argument("--mutation_rate_step", type=float, default=1.2,
                    help="mutation rate step unit")
# experiment replated params
parser.add_argument("--num_iter", type=int, default=100,
                    help="number of iterations for one experiment")
parser.add_argument("--num_proc", type=int, default=1,
                    help="number of processes for experiment")
# dir params
parser.add_argument("--input", default=None,
                    help="path to the saved Experiment model")
parser.add_argument("--output", default=None,
                    help="path for Experiment to be saved")
parser.add_argument("--model_name", default="CoalSim.model",
                    help="filename of the Experiment model to be saved")
# flags
parser.add_argument("--verbose", action="store_true", help="increase verbosity")

def get_config():
  config, _ = parser.parse_known_args()
  return config
