# -*- coding: utf-8 -*-

import numpy as np

from models import T

__author__ = 'Jayeol Chun'

def project_onto_plane(a, b):
    """
    finds the vector projection of points onto the hyperplane
    a : coefficients of the hyperplane
    b : original vector
    return  : new vector projected onto the hyperplane
    """
    dot = np.dot(a, b) / np.linalg.norm(a)
    p = dot * a / np.linalg.norm(a)
    return b - p


def display_params(args):
    """
    displays the simulation's parameber values
    """
    print("\n******* Current Simulation Params *******")
    print('\tSample Size: {}'.format(args[0]))
    print('\tMutation Rate: {:.3f}'.format(args[1]))


def display_stats(data):
    """
    displays the cumulative statistics of all trees observed for the models
    """
    k, b = data
    print("\n<< Kingman vs. Bolthausen-Sznitman Stats >>")
    print("\t{}:\t{:.2f} vs {:.2f}".format("AVG", np.mean(k), np.mean(b)))
    print("\t{}:\t{:.2f} vs {:.2f}".format("VAR", np.var(k), np.var(b)))




