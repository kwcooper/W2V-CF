# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:36:11 2018

@author: prudh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  9 00:57:28 2018

@author: prudh
"""

#import tensorflow as tf
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib auto
import pickle
import collections
import string
import gensim
stopwords = open('stopwords_en.txt').read().split() #missing this file

def fetch_clean_data(data):    
    # split sentences
    sentences = []
    for d in data:
        for sent in d:
            sentences.append(sent)
    # remove '[S]' tag from the sentences
    clean_sentences = ' '
    for s in sentences:
        if '[S]' in s: 
            clean_sentences += ' ' + s[4:]
            
    # remove punctuations
    table = str.maketrans({key: None for key in string.punctuation})
    new_s = clean_sentences.translate(table)    
    
    # remove numbers
    no_numbers = ''.join(i for i in new_s if not i.isdigit())
    
    # remove stop words
    remove_stop = ' '
    tokens = no_numbers.split()
    for n in tokens:
        if n not in stopwords:
            remove_stop += n + ' '
    
    # collect unique vocabulary
    vocab = np.unique(remove_stop.split())
    
    # remove words occuring just once
    cnt = collections.Counter()
    no_stop_tokes = remove_stop.split()
    for word in no_stop_tokes:
        cnt[word] +=1
    
    filtered_dict = {}
    for c in cnt:
        if cnt[c] > 1:
            filtered_dict[c] = cnt[c]
    
    # generate final text for processing
    wor = remove_stop.split()
    text = ' ' 
    for r in wor:
        if r in filtered_dict:
            text += r + ' ' 
    
    data = text.split()
    return data


# seperate sentences with a window of size 5
def seperate_sentences(topic_data):
    sep_sent = []
    for d in range(0, len(topic_data)-11):
        sep_sent.append(topic_data[d : d+11])
    return sep_sent


# topic 1 sense (dominant)
fname1 = open('tasa_corpus/industrailarts.txt', 'rb')
data1 = pickle.load(fname1)

# topic 2 sense (subordinate)
fname2 = open('tasa_corpus/socialstudies.txt', 'rb')
data2 = pickle.load(fname2)

# collect topic 1 sentences
topic1_sent = seperate_sentences(fetch_clean_data(data1))
# collect topic 2 sentences
topic2_sent = seperate_sentences(fetch_clean_data(data2))

#Random ordering
#combine = topic2_sent + topic1_sent
#np.random.shuffle(combine)

# topic 1 - topic 2
#combine = topic1_sent + topic2_sent

# topic 2 - topic 1
#combine = topic2_sent + topic1_sent

# create similairty dictionary
sim_dict = defaultdict(dict)

# target word (homonym)
target_word = 'plane'

# comparison words (words synonymous to target work in two topics)
compare_words= ['board', 'speedboat']

# initialize similarity dictionary with empty lists 
sim_dict[target_word][compare_words[0]] = []
sim_dict[target_word][compare_words[1]] = []

number_of_runs = 10
# sg - skipgram
# alpha - learning rate
# size - embedding size
# window - context window
# seed - random number seed
# min_count - words less than min_count are disgarded
# negative - number of negative samples drawn
# iter - number of iterations
for i in range(0, number_of_runs):
    print (i)
    model = gensim.models.word2vec.Word2Vec(combine, sg = 1, 
                                            alpha = 0.16,
                                            size = 50,
                                            window = 5,
                                            seed = i,
                                            min_count = 0,
                                            hs = 0,
                                            negative = 1,
                                            iter = 1)   

# save similarity between target and comparison word one
    sim_dict[target_word][compare_words[0]].append(cosine_similarity([model.wv[target_word]], [model.wv[compare_words[0]]])[0][0])
# save similarity between target and comparison word two
    sim_dict[target_word][compare_words[1]].append(cosine_similarity([model.wv[target_word]], [model.wv[compare_words[1]]])[0][0])


print ('Mean')
print (target_word + ' - ' + compare_words[0] + ' :', np.mean(sim_dict[target_word][compare_words[0]])) 
print (target_word + ' - ' + compare_words[1] + ' :', np.mean(sim_dict[target_word][compare_words[1]])) 
print ('Standard Deviation')
print (target_word + ' - ' + compare_words[0] + ' :', np.std(sim_dict[target_word][compare_words[0]])) 
print (target_word + ' - ' + compare_words[1] + ' :', np.std(sim_dict[target_word][compare_words[1]])) 

dframe = pd.DataFrame(sim_dict)
dframe.to_pickle('<name_to_save>.pkl')
    
    
