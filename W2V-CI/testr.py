#testr

import os
from collections import defaultdict

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
# legacy from handeling unequal lists
senVeh = []
senDish = []
for sent in data:
    if sent.split()[2] == 'car' or sent.split()[2] == 'truck':
        senVeh.append(sent)
    elif sent.split()[2] == 'glass' or sent.split()[2] == 'plate':
        senDish.append(sent)

##slic = int(len(senVeh) - len(senDish)/2)
##dataa = senDish + senVeh[slic:]

dishVeh = senDish + senVeh

# break the sentences up into lists of words
sentences = []
for s in dishVeh:
    sentences.append(s.split())

# collect each of the words used in the sentenses
vocab = []
for s in dishVeh:
    for si in s.split():
        if si not in vocab:
            vocab.append(si)
   
def return_vectors(model, vocab):
    vectorDict = {}
    for v in vocab:
        #print v
        vectorDict[v] = model.wv[v]
    return vectorDict

def save_vectors(vector_dict, i):
    filename = open("vectors/veh-dish_vectors_" +str(i) + ".pkl", "wb")
    pickle.dump(vector_dict, filename)

# Train the W2V model!
vectorDic = defaultdict(dict)
iterations = 100
print('Training Model...')
for i in range(0, iterations):
    #np.random.shuffle(sentences)
    if (i < 100 & i % 10 ==0) or i % 100 == 0:
        print('iteration: ', i)
    # uses skipgram, 300 dementions, max dist 2, 5 iterations, seed changes
    model = gensim.models.Word2Vec(sentences, sg=1, size=300, window=2, iter=5, seed=i)
    vectors = return_vectors(model, vocab)
    vectorDic[i] = vectors 

