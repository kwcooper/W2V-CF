# vocab functions
# currently only accepts hardcoded corpus 

import numpy as np
from collections import defaultdict


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
