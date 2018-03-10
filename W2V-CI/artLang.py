
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
order = 0 # Determines the order of sentenses [0 1]
shuff = False # For random condition
iterations = 10 # 10k~15m; 1k~1.5m


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
if order:
    tenses = senDish + senVeh
if ~order:
    tenses = senVeh + senDish

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
    # uses skipgram, 300 dimensions, max dist 2, 5 iterations, seed changes
    model = gensim.models.Word2Vec(sentences, sg=1, size=300, window=2, iter=5, seed=i)
    vectors = returnVectors(model, vocab)
    vectorDic[i] = vectors
    trainingTime.append((time.time()-start))
    
print("Time:", (time.time() - start)/60, "minutes")

# Now let"s compute the distances to the queryWord
full = copy.deepcopy(vectorDic)
queryWord=  "break"
checkWord = ["car", "truck", "glass", "plate"]
cosDic = defaultdict(dict)
for i in range(0, iterations):
    first = full[i]
    for word in checkWord:
        cosDic[i][word] = cosine(first[queryWord], first[word])


# Compute final measurements
print("\nResults:")
df = pd.DataFrame(cosDic).T

df["Vehicles"] = (df["car"] + df["truck"])/2
df["Dinnerware"] = (df["glass"] + df["plate"])/2

# for how many is the distance greater
df["closer2vehicles"] = (df["Vehicles"] < df["Dinnerware"])
df["closer2dinnerware"] = (df["Dinnerware"] < df["Vehicles"])

# Display results
print("Vehicles Occuring First: " + str(sum(df["closer2vehicles"])))
print("Dinnerware Occuring First: " + str(sum(df["closer2dinnerware"])))
print("Proportion of Vehicles Occuring First: " + str(sum(df["closer2vehicles"])/float(iterations)))
print("Proportion of Dinnerware Occuring First: " + str(sum(df["closer2dinnerware"])/float(iterations)))

c2v = sum(df["closer2vehicles"])/float(iterations)
c2d = sum(df["closer2dinnerware"])/float(iterations)
print("v to d Ratio:", c2v/c2d)

name = "results/VehicleDinnerware_" + str(iterations) + "runs.csv"
print("\nSaving to", name)
df.to_csv(name)



#plotting results
print("Plotting results")
distVeh = df["Vehicles"]
distDish = df["Dinnerware"]
distVehMean = np.mean(distVeh)
distDishMean = np.mean(distDish)

labels = ("Vehicles", "Dinnerware")
yPos = np.arange(len(labels))
results = [distVehMean, distDishMean]
 
plt.bar(yPos, results, align="center", alpha=0.5)
plt.xticks(yPos, labels)
plt.ylabel("Distance")
t = "Distance from cue word | " + str(iterations) + " Iterations"
plt.title(t)
plt.show()

# training time plt
plt.plot(trainingTime)
plt.title("Training time (s per iteration)")
plt.show()

print("\nfin")
