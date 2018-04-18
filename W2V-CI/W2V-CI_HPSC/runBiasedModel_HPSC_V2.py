
import os
import copy
import time
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

import gensim

# td
#   add functions for this and model training
#   add logging dataframe


#4 sims:  2d1v; 1v2d; 2v1d; 1d2v
print("\nWord2Vec Catastrophic Forgeting Analysis")
# simulation parameters
#order = 0 # Determines the order of sentenses 1:Db4V 0:Vb4D

 
simResults = []
for i in [3]:

    shuff = False # For random condition
    iterations = 10000 # 10k~15m; 1k~1.5m
    #2d1v; 1v2d; 2v1d; 1d2v
    comb = i
    
    # Grab the data
    os.chdir("corpus")
    print(os.getcwd())
    d1 = open("biasArtLang_8xVeh-667_Dish-1333.txt").read().splitlines()
    d2 = open("biasArtLang_8xVeh-1333_Dish-667.txt").read().splitlines()
    os.chdir("..")


    def returnVectors(model, vocab):
        vectorDict = {}
        for v in vocab:
            vectorDict[v] = model.wv[v]
        return vectorDict

    def saveVectors(vector_dict, i):
        filename = open("vectors/veh-dish_vectors_" +str(i) + ".pkl", "wb")
        pickle.dump(vector_dict, filename)

    # choose combination
    if comb == 1:
        data = d1
        order = 1
        rTxt = "2d1v"
    elif comb == 2:
        data = d1
        order = 0
        rTxt = "1v2d"
    elif comb == 3:
        data = d2
        order = 0
        rTxt = "2v1d"
    elif comb == 4:
        data = d2
        order = 1
        rTxt = "1d2v"
    else:
        print("Error")

    if shuff:
        rTxt = rTxt + "-Rand"
    print("Running", rTxt)



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

    print("Running", oTxt[0:6])

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
    print("\nTraining the Model...")
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
    print("\nRESULTS:")
    df = pd.DataFrame(cosDic).T

    #print(df.head())
    # find average distance
    df["Vehicles"] = (df["car"] + df["truck"])/2
    df["Dinnerware"] = (df["glass"] + df["plate"])/2

    # for how many is the distance greater
    df["closer2vehicles"] = (df["Vehicles"] < df["Dinnerware"])
    df["closer2dinnerware"] = (df["Dinnerware"] < df["Vehicles"])

    # Display results

    print("Vehicles Occuring First: " + str(sum(df["closer2vehicles"])))
    print("Dinnerware Occuring First: " + str(sum(df["closer2dinnerware"])))

    c2v = sum(df["closer2vehicles"])/float(iterations)
    c2d = sum(df["closer2dinnerware"])/float(iterations)
    print("v to d Ratio:", c2v/c2d)

    name = "results/biased/" + rTxt + "_" + str(iterations) + "runs.csv"
    print("\nSaving to", name)
    df.to_csv(name)



    # binary rank results
    c2v = sum(df["closer2vehicles"])
    c2d = sum(df["closer2dinnerware"])
    print("Binary Rank: c2v:", c2v,"c2d:", c2d)

    # distance results
    distVeh = df["Vehicles"]
    distDish = df["Dinnerware"]
    distVehMean = np.mean(distVeh)
    distDishMean = np.mean(distDish)
    print("Distance: VehMean:", distVehMean, "DishMean:", distDishMean)

    simResults.append([rTxt, {'br_c2v-c2d':[c2v, c2d], 'dist_vM-dM':[distVehMean, distDishMean]}])
    
print("\nfin")

print(simResults)
