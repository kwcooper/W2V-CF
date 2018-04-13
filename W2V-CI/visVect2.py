
from sklearn.manifold import TSNE


import os
import copy
import time
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

import gensim

# td
#   add functions for this and model training
#   add logging dataframe

# Grab the data
os.chdir("corpus")
print(os.getcwd())
data = open("artLang_2s_8x1000_shuffled.txt").read().splitlines()
os.chdir("..")

# simulation parameters
order = 1 # Determines the order of sentenses 1:Db4V 0:Vb4D
shuff = False # For random condition
iterations = 100 # 10k~15m; 1k~1.5m


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
    if sent.split()[2] == "car" or sent.split()[2] == "truck":
        senVeh.append(sent)
    elif sent.split()[2] == "glass" or sent.split()[2] == "plate":
        senDish.append(sent)

##slic = int(len(senVeh) - len(senDish)/2)
##dataa = senDish + senVeh[slic:]
        
# determine the training order
if order == 1:
    tenses = senDish + senVeh
    oTxt = "D b4 V |"
else:
    tenses = senVeh + senDish
    oTxt = "V b4 D |"

# break the sentences up into lists of words
sentences = []
for s in tenses:
    sentences.append(s.split())

# collect each of the words used in the sentenses
vocab = []
for s in tenses:
    for si in s.split():
        if si not in vocab:
            vocab.append(si)
   

# Train the W2V model!
vectorDic = defaultdict(dict)
print("Training the Model...")
trainingTime = []
start = time.time()
for i in range(0, iterations):
    if shuff:
        np.random.shuffle(sentences)
    #if (i <= 100 and i % 10 == 0) or i % 1000 == 0:
    if i % 1000 == 0:
        print("iteration: ", i)
    #reduce  iter to test overfitting?
    # uses skipgram, 300 dimensions, max dist 2, 5 iterations, seed changes
    model = gensim.models.Word2Vec(sentences, sg=1, size=300, window=2, iter=2, seed=i)
    vectors = returnVectors(model, vocab)
    vectorDic[i] = vectors
    trainingTime.append((time.time()-start))
    
print("Time:", (time.time() - start)/60, "minutes")


print(model['break']) 
model.similar_by_word('break')

def display_closestwords_tsnescatterplot(model, word):
    arr = np.empty((0,300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()


display_closestwords_tsnescatterplot(model, 'break')
