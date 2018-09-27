# vocab functions
# currently only excepts hardcoded corpus 

import numpy as np
from collections import defaultdict

# pass corpus to the class, then magic happens and you get data
class Corpus:
    def __init__(self): # should pass sentences to this. 
        # many of these to be moved out and refactored
        self.fish_data = ['bass fish',
                         'bass eat', 
                         'trout fish',
                         'trout eat',]
        self.instrument_data = ['bass play',
                               'bass pluck',
                               'acoustic play',
                               'acoustic pluck']
        self.vocab = ['bass', 'acoustic', 'trout', 'fish', 'eat', 'play', 'pluck']
        self.sent_reps = 1000
        self.order = 'rand' # rand, aLast, fLast
        self.dom = [0,0] # fish, acoustic
        self.generate_corpus = []
        self.test = []
        self.train = []

        #1000, False, False, 20, 'random')
    def fetchData(self, dom, order, sent_reps):
        self.sent_reps = sent_reps
        self.order = order # rand, aLast, fLast
        fish = self.sent_hierarchy(self.fish_data, self.dom[0])
        instrument = self.sent_hierarchy(self.instrument_data, self.dom[1])
        #shuffle each sense
        np.random.shuffle(fish)
        np.random.shuffle(instrument)
        if order == 'rand':
            generate_corpus = instrument + fish
            np.random.shuffle(generate_corpus)
        elif order == 'aLast':
           generate_corpus = fish + instrument
        elif order == 'fLast':
           generate_corpus = instrument + fish

        self.generate_corpus = generate_corpus
        self.makeWordIndexing()

        # create input and output feeds (Train and test?)
        input_feed = []
        output_feed = []
        for c in generate_corpus:
            splitted = c.split()
            input_feed.append(self.word2dex[splitted[0]])
            output_feed.append(self.word2dex[splitted[1]]) #W: make sure to uncomment case code, otherwise generate_corpus uses old code that includes

        input_feed = np.array(input_feed)
        output_feed = np.reshape(np.array(output_feed), (len(output_feed), 1))
        #feed_dict = {train_inputs: [input_feed[e]], train_labels: [output_feed[e]]}

        self.test = input_feed
        self.train = output_feed
        return self.test, self.train
    
    def getTrainingSet(self):
        pass

    def getVocab(self):
    # get vocab list from corpus
        pass

    def sent_hierarchy(self, data, dominant):
    # generates dominant and subordinate repetitions
    # called by fetch data
        inner_list = []
        if self.dom :
            for d in data:
                inner_list += [d] * self.sent_reps
        if dominant == False:
            for d in data:
                # scale by some factor
                # this can be altered to test effects on forgetting
                inner_list += [d] * (int(self.sent_reps/3))
        return inner_list
    
    def makeWordIndexing(self):
        word2dex, dex2word = {}, {}
        for v in range(0, len(self.vocab)):
            word2dex[self.vocab[v]] = v
            dex2word[v] = self.vocab[v]
        self.word2dex = word2dex
        self.dex2word = dex2word

        
    def makeSimDict(self):
        # initialize the similarity dictionary
        # move outside code
        for v in self.vocab:
            for ve in self.vocab:
                sim_dict[v][ve] = []

        # create a 2d-dictionary with list
        sim_dict = defaultdict(dict)
        self.sim_dict = sim_dict
        return simDict


if __name__ == "__main__":
    c = Corpus()
            #1000, False, False, 20, 'random')
    tst,trn = c.fetchData([0,0], 'rand', 10)

    print(c.word2dex[c.vocab[0]])
    if 0:
        for i,j in zip(tst,trn):
            print(i,j)













# switch 1, 2, 3
def generateCorpus(corpus, n_sent_reps, switch):

    

    # replicating each sentence n_sent_reps times
    bass_fish = [corpus[0]] * n_sent_reps    
    trout_fish = [corpus[1]] * n_sent_reps    
    bass_guitar = [corpus[2]] * n_sent_reps    
    acoustic_guitar = [corpus[3]] * n_sent_reps    

    # randomize difference sense sentences
    diff_sentences = trout_fish + acoustic_guitar
    np.random.shuffle(diff_sentences)

    #* add them equally to both the senses
    fish_sense = bass_fish + diff_sentences[0: int(len(diff_sentences)/2)]
    instrument_sense = bass_guitar + diff_sentences[int(len(diff_sentences)/2):]
    #fish_sense = bass_fish + trout_fish
    #instrument_sense = bass_guitar + acoustic_guitar

    #* shuffle the sentences in each sense
    np.random.shuffle(fish_sense)
    np.random.shuffle(instrument_sense)

    # Determine sentense ordering

    if switch == 1:
        # add fish + instrument for FI ordering
        generate_corpus = fish_sense + instrument_sense
        print('Order: FI')
    elif switch == 2:
        # add instrument + fish for IF ordering
        generate_corpus = instrument_sense + fish_sense
        print('Order: IF')
    else:
        # random ordering
        generate_corpus = fish_sense + instrument_sense
        np.random.shuffle(generate_corpus)
        print('Order: RAND')

    # Grab indicies
    vocab = ['bass', 'guitar', 'acoustic', 'trout', 'fish']
    word_to_index, index_to_word = {}, {}
    for v in range(0, len(vocab)):
        word_to_index[vocab[v]] = v
        index_to_word[v] = vocab[v]
        print('{}:{}'.format(v,vocab[v]), end=" ")
    print()


    # Depriciated? Define similarity matrix
    tot_dict, sim_dict1 = defaultdict(dict), defaultdict(dict)
    for v in vocab:
        for ve in vocab:
            tot_dict[v][ve] = 0.
    #        sim_dict1[v][ve] = 0.


    #np.random.shuffle(total_corpus)
    #np.random.shuffle(generate_corpus)
    #np.random.shuffle(fish_sense)
    #np.random.shuffle(instrument_sense)

    # add fish + instrument for FI ordering
    #generate_corpus = instrument_sense + fish_sense

    # add instrument + fish for IF ordering
    #generate_corpus = instrument_sense + fish_sense

    # random ordering
    #generate_corpus = instrument_sense + fish_sense
    #np.random.shuffle(generate_corpus)

    # generate input and output lists of vocab
    input_feed, output_feed = [], []
    for c in generate_corpus:
        splitted = c.split()
        #print(splitted)
        input_feed.append(word_to_index[splitted[0]])
        output_feed.append(word_to_index[splitted[1]])
    input_feed = np.array(input_feed)
    output_feed = np.reshape(np.array(output_feed), (len(output_feed), 1))
    
    return input_feed, output_feed, generate_corpus, word_to_index, index_to_word, vocab
