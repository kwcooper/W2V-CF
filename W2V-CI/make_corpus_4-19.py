#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 19:16:04 2018

@author: prudhvi
"""

import numpy as np

# Open data file
collecter = open('corpus.txt', 'r').read().splitlines()

appender = []
half_appender = []
for c in collecter:
    # duplicate each sentence 500 times
    appender += [c] * 500
    # duplicate each sentence 250 times
    half_appender += [c] * 250


def data_splitter(data):
    # two lists for capturing both senses
    vehicles = []
    dinnerware = []
    # for each sample
    for c in data:
        # if it has car or truck, put it in vehicles
        if c.split()[2] == 'car' or c.split()[2] == 'truck':
            vehicles.append(c)
        # if it has glass or plate, put it in dinnerware
        elif c.split()[2] == 'glass' or c.split()[2] == 'plate':
            dinnerware.append(c)
    # return both lists
    return vehicles, dinnerware

# retreive full data
vehicles, dinnerware = data_splitter(appender)

# retrive half data
half_vehicles, half_dinnerware = data_splitter(half_appender)

'''
Equal Senses
Random  -> np.random.shuffle(vehicles + dinnerware)
1V1D -> vehicles + dinnerware
1D1V -> dinnerware + vehicles

Unequal Senses
Random(2D1V) -> np.random.shuffle(dinnerware + half_vehicles)
Random(2V1D) -> np.random.shuffle(vehicles + half_dinnerware)
2D1V -> dinnerware + half_vehicles
2V1D -> vehicles + half_dinnerware
'''