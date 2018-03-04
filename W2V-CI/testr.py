#testr

import os
import numpy as np
import pickle

from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

import gensim

import matplotlib.pyplot as plt


# Grab the data
os.chdir("corpus")
print(os.getcwd())
data = open('artLang_2s_8x1000_shuffled.txt').read().splitlines()
os.chdir("..")

print(data[1:10])

# Split the lists into their respective senses
# td: this whole section and above can be replaced 
# by importing the makeArtLangCorpus function...
senVeh = []
senDish =  []
for sent in data:
    if sent.split()[2] == 'car' or sent.split()[2] == 'truck':
        senVeh.append(sent)
    elif sent.split()[2] == 'glass' or sent.split()[2] == 'plate':
        senDish.append(sent)

# legacy from handeling unequal lists
##slic = int(len(senVeh) - len(senDish)/2)
##dataa = senDish + senVeh[slic:]

dataa = senDish + senVeh

sentences = []
for d in dataa:
    sentences.append(d.split())
