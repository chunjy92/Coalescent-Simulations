#! /usr/bin/python
# -*- coding: utf-8 -*-
from .Kingman import Kingman
from .BolthausenSznitman import BolthausenSznitman
from .nodes import Sample, Ancestor

__author__ = 'Jayeol Chun'


MODELS = [Kingman(), BolthausenSznitman()]
