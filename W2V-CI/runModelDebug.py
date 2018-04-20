
import os
import copy
import time
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

import gensim
from gensim.test.utils import common_corpus, common_dictionary
from gensim.similarities import MatrixSimilarity

# td
#   add functions for this and model training
#   add logging dataframe
#   mds Plot

# Grab the data
os.chdir("corpus")
print(os.getcwd())
data = open("artLang-8000_1000-8s-1t_hom.txt").read().splitlines()
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

def getSimilarityMatrix(model, vocab):
    print(gensim.similarities)
    SM = np.zeros((len(vocab), len(vocab)))
    for i in range(len(vocab)):
        for j in range(len(vocab)):
            SM[i,j] = model.wv.similarity(vocab[i],vocab[j])
    return SM
            

def getSimilarityMatrix2(model, vocab):
    SM = MatrixSimilarity(common_corpus, num_features=len(common_dictionary))
    print(SM[[(1, 2), (5, 4)]])
    

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


#test the random assignment
##randomList = np.random.randint(0, len(senVeh), int(len(senDish)/2))
##tenses = senDish + list(np.array(senVeh)[randomList])


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
print('vocab:', vocab)

# Train the W2V model!
vectorDic = defaultdict(dict)
print("Training the Model...")
trainingTime = []
SML = []
start = time.time()
itr = 3
for i in range(0, iterations):
    if shuff:
        np.random.shuffle(sentences)
    #if (i <= 100 and i % 10 == 0) or i % 1000 == 0:
    if i % 1000 == 0:
        print("iteration: ", i)
    #reduce  iter to test overfitting?
    # uses skipgram, 300 dimensions, max dist 2, 5 iterations, seed changes
    model = gensim.models.Word2Vec(sentences, sg=0, size=300, window=2, iter=itr, seed=i)
    vectors = returnVectors(model, vocab)
    #print(i)
    #SML.append(getSimilarityMatrix(model, vocab))
    vectorDic[i] = vectors
    trainingTime.append((time.time()-start))
    
print("Time:", (time.time() - start)/60, "minutes")


from collections import defaultdict
simMat = defaultdict(dict) # a dictionary of dictionaries
for v in vectors:
    for ve in vectors:
        simMat[v][ve] = model.wv.similarity(v,ve)

dframe = pd.DataFrame(simMat)
t = "SM_Vehicles_itr" + str(itr) + "_it" + str(iterations) + ".csv"
dframe.to_csv(t)
print("saved", t)

input()
# Now let"s compute the distances to the queryWord
full = copy.deepcopy(vectorDic)
queryWord = "break"
checkWord = ["stop", "smash"]
cosDic = defaultdict(dict)
for i in range(0, iterations):
    first = full[i]
    for word in checkWord:
       cosDic[i][word] = cosine(first[queryWord], first[word])
        #cosDic[i][word] = cosine_similarity(first[queryWord], first[word])
        
#SMA = np.mean(np.array([ old_set, new_set ]), axis=0 )




# Compute final measurements
print("\nResults:")
df = pd.DataFrame(cosDic).T

# calculate the distance
df["Vehicles"] = df["stop"]
df["Dinnerware"] = df["smash"]

# for how many is the distance greater
df["closer2vehicles"] = (df["stop"] < df["smash"])
df["closer2dinnerware"] = (df["smash"] < df["stop"])

# Display results
print("Vehicles Occuring First: " + str(sum(df["closer2vehicles"])))
print("Dinnerware Occuring First: " + str(sum(df["closer2dinnerware"])))

c2v = sum(df["closer2vehicles"])/float(iterations)
c2d = sum(df["closer2dinnerware"])/float(iterations)

name = "results/VehicleDinnerware_2iter_" + str(iterations) + "runs.csv"
print("\nSaving to", name)
df.to_csv(name)



# Binary Rank 
print("Plotting results")
c2v = sum(df["closer2vehicles"])
c2d = sum(df["closer2dinnerware"])
print(c2v,c2d)
labels = ("c2Vehicles", "c2Dinnerware")
yPos = np.arange(len(labels))
results = [c2v, c2d]
 
plt.bar(yPos, results, align="center", alpha=0.5)
plt.xticks(yPos, labels)
plt.ylabel("Iterations")
t = "2 iter Binary Rank | " + oTxt + str(iterations) + " Iterations"
plt.title(t)
plt.show()




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
t = "2iter Distance from cue word | " + str(iterations) + " Iterations"
plt.title(t)
plt.show()

# training time plt
##plt.plot(trainingTime)
##plt.title("Training time (s per iteration)")
##plt.show()

print("\nfin")
