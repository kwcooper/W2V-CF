#testr

import os
import copy
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

import gensim


# Grab the data
os.chdir("corpus")
print(os.getcwd())
data = open('artLang_2s_8x1000_shuffled.txt').read().splitlines()
os.chdir("..")

def returnVectors(model, vocab):
    vectorDict = {}
    for v in vocab:
        vectorDict[v] = model.wv[v]
    return vectorDict

def saveVectors(vector_dict, i):
    filename = open("vectors/veh-dish_vectors_" +str(i) + ".pkl", "wb")
    pickle.dump(vector_dict, filename)

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
        
# This determines the training order!
dishVeh =  senVeh + senDish

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
   

# Train the W2V model!
vectorDic = defaultdict(dict)
iterations = 100
print("Training the Model...")
for i in range(0, iterations):
    #np.random.shuffle(sentences)
    if (i <= 100 and i % 10 == 0) or i % 100 == 0:
        print("iteration: ", i)
    # uses skipgram, 300 dimensions, max dist 2, 5 iterations, seed changes
    model = gensim.models.Word2Vec(sentences, sg=1, size=300, window=2, iter=5, seed=i)
    vectors = returnVectors(model, vocab)
    vectorDic[i] = vectors 

# Now let's compute the distances
full = copy.deepcopy(vectorDic)
queryWord=  'break'
checkWord = ['car', 'truck', 'glass', 'plate']
cosDic = defaultdict(dict)
for i in range(0, iterations):
    first = full[i]
    for word in checkWord:
        cosDic[i][word] = cosine(first[queryWord], first[word])


# Compute final measurements
print("\nResults:")
dframe = pd.DataFrame(cosDic).T

dframe['Vehicles'] = (dframe['car'] + dframe['truck'])/2
dframe['Dinnerware'] = (dframe['glass'] + dframe['plate'])/2

dframe['closer2vehicles'] = (dframe['Vehicles'] < dframe['Dinnerware'])
dframe['closer2dinnerware'] = (dframe['Dinnerware'] < dframe['Vehicles'])

# Display results
print("Vehicles Occuring First: " + str(sum(dframe['closer2vehicles'])))
print("Dinnerware Occuring First: " + str(sum(dframe['closer2dinnerware'])))
print("Proportion of Vehicles Occuring First: " + str(sum(dframe['closer2vehicles'])/float(iterations)))
print("Proportion of Dinnerware Occuring First: " + str(sum(dframe['closer2dinnerware'])/float(iterations)))

#name = ''
#dframe.to_csv('results/VehicleDinnerware_100Runs.csv')
