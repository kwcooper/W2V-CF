import numpy as np
import os
import pickle

from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
#os.chdir("Desktop")

#for i in range(0, 100):
import copy
import pandas as pd




dframe['Vehicles'] = (dframe['car'] + dframe['truck'])/2
dframe['Dinnerware'] = (dframe['glass'] + dframe['plate'])/2
#dframe['News'] = (dframe['news'] + dframe['story'])/2


dframe['closer to vehicles'] = (dframe['Vehicles'] < dframe['Dinnerware']) #& (dframe['Vehicles'] < dframe['News']) 
dframe['closer to dinnerware'] = (dframe['Dinnerware'] < dframe['Vehicles'])# & (dframe['Dinnerware'] < dframe['News']) 

#dframe['closer to vehicles'] = (dframe['Vehicles'] < dframe['Dinnerware']) & (dframe['Vehicles'] < dframe['News']) 
#dframe['closer to dinnerware'] = (dframe['Dinnerware'] < dframe['Vehicles']) & (dframe['Dinnerware'] < dframe['News']) 
#dframe['closer to news'] = (dframe['News'] < dframe['Dinnerware']) & (dframe['News'] < dframe['Vehicles']) 

print("Vehicles Occuring First: " + str(sum(dframe['closer to vehicles'])))
print( "Dinnerware Occuring First: " + str(sum(dframe['closer to dinnerware'])))
#print "News Occuring First: " + str(sum(dframe['closer to news']))

print( "Proportion of Vehicles Occuring First: " + str(sum(dframe['closer to vehicles'])/10000.0))
print( "Proportion of Dinnerware Occuring First: " + str(sum(dframe['closer to dinnerware'])/10000.0))
#print "Phroportion of News Occuring First: " + str(sum(dframe['closer to news'])/10000.0))

dframe.to_csv('VehicleDinnerware 10000 Runs Unbal dinner_vehi1.csv')
