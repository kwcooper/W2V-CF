# w2v-ci tensorflow implimentation

import tensorflow as tf
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib auto
#import gensim
import time

import vocabFunctions as vp

# list of 4 sentences
corpus = ['bass fish', 'trout fish', 'bass guitar', 'acoustic guitar']
    
# Grab the corpus data
# FI = 1, IF = 2, RAND = ~3
switch = 1
input_feed, output_feed, generate_corpus, word_to_index, index_to_word, vocab = vp.generateCorpus(corpus, 1000, switch)


# Run experiment n times
tot_bass_trout = []
tot_bass_acoustic = []
start = time.time()
num_runs = 5
embedding_size = 10
print("Embeddings size = {}x{}".format(len(vocab), embedding_size))
print('Run ', end="")
for i in range(0, num_runs):
    print('{}...'.format(i+1), end=" ")

    
    # construct network
    tf.reset_default_graph() 
    embeddings = tf.Variable(tf.random_uniform([len(vocab), embedding_size], -1.0, 1.0)) # 5 X 10
    
    nce_weights = tf.Variable(tf.truncated_normal([len(vocab), embedding_size], stddev=1.0/np.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([len(vocab)])) #5
    
    train_inputs = tf.placeholder(tf.int32, shape=[1])
    train_labels = tf.placeholder(tf.int32, shape=[1,1])
    
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    #out_embed = tf.nn.embedding_lookup(nce_weights, train_inputs)

    # reduce the mean of  the noise-contrastive estimation training loss
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=train_labels,
                                         inputs=embed,
                                         num_sampled=1,
                                         num_classes=len(vocab)))
    
    # Train by reducing the calculated loss
    optimizer = tf.train.AdamOptimizer(learning_rate=0.02).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

    
    # Run the model
    sim_dict = defaultdict(dict)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    bass_fish_loss, bass_guitar_loss = [], []
    for e in range(0, len(input_feed)):
        x = sess.run(optimizer, feed_dict={train_inputs:[input_feed[e]], train_labels:[output_feed[e]]})

        # Grab embedding vector for each word
        inp_vectors = {}
        for v in range(0, len(vocab)):
            inp_vectors[vocab[v]] = sess.run(embed, feed_dict={train_inputs:[word_to_index[vocab[v]]]})
            #print(v, inp_vectors[vocab[v]])

        # calculate the cosine similarity between vectors
        for v in inp_vectors:
            for vv in inp_vectors:
                sim_dict[v][vv] = cosine_similarity(inp_vectors[v], inp_vectors[vv])[0][0]

        bass_fish_loss.append(sim_dict['bass']['trout'])
        bass_guitar_loss.append(sim_dict['bass']['acoustic'])
    tot_bass_trout.append(bass_fish_loss)
    tot_bass_acoustic.append(bass_guitar_loss)

print('Took {} seconds'.format((start-time.time())))
mean_bass_trout = np.mean(tot_bass_trout, axis=0)
mean_bass_acoustic = np.mean(tot_bass_acoustic, axis=0)

# PLOTTING
# add legend to plots.
print('\nPlotting and saving figs...')
plt.figure(figsize=(25, 15))
plt.xlabel('Samples')
plt.ylabel('Cosine')
plt.ylim(0, 1)   
plt.scatter(np.arange(0, len(mean_bass_trout), 1), mean_bass_trout, color='red')
plt.scatter(np.arange(0, len(mean_bass_acoustic), 1), mean_bass_acoustic, color='green')
if switch == 1:
    plt.title('FI Ordering: Average of 5 Runs - 1000 Samples Each \n GREEN: Cosine between bass and acoustic (Guitar sense) \n RED: Cosine between bass and trout (Fish sense)')
    plt.savefig('FI_1000samp_adam.png', dpi=300)
elif switch == 2:
    plt.title('IF Ordering: Average of 5 Runs - 1000 Samples Each \n GREEN: Cosine between bass and acoustic (Guitar sense) \n RED: Cosine between bass and trout (Fish sense)')
    plt.savefig('IF_1000samp_adam.png', dpi=300)    
else:
    plt.title('Random Ordering: Average of 5 Runs - 1000 Samples Each \n GREEN: Cosine between bass and acoustic (Guitar sense) \n RED: Cosine between bass and trout (Fish sense)')
    plt.savefig('Random_1000samp_adam.png', dpi=300)    



print('\nfin')

