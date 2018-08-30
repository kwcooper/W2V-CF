# testr

##vocab = ['bass', 'guitar', 'acoustic', 'trout', 'fish']
##word_to_index, index_to_word = {}, {}
##for v in range(0, len(vocab)):
##    word_to_index[vocab[v]] = v
##    index_to_word[v] = vocab[v]
##    print('{}:{}'.format(v,vocab[v]), end=" ")
##print()
##
##corpp = {'bass': 0, 'guitar': 1, 'acoustic': 2, 'trout': 3, 'fish': 4}
##corpus = ['bass fish', 'trout fish', 'bass guitar', 'acoustic guitar']
##sentList = [[0,4],[3,4],[0,1],[2,1]]
##inpts = [0,3,0,2]
##outputs = [4,4,1,1]
##
##bass_fish = [sentList[0]] * n_sent_reps    
##trout_fish = [sentList[1]] * n_sent_reps    
##bass_guitar = [sentList[2]] * n_sent_reps    
##acoustic_guitar = [sentList[3]] * n_sent_reps 
##
### randomize difference sense sentences
##diff_sentences = trout_fish + acoustic_guitar
##np.random.shuffle(diff_sentences)
##
###* add them equally to both the senses
##fish_sense = bass_fish + diff_sentences[0: int(len(diff_sentences)/2)]
##instrument_sense = bass_guitar + diff_sentences[int(len(diff_sentences)/2):]


import vocabFunctions as vp

# list of 4 sentences
corpus = ['bass fish', 'trout fish', 'bass guitar', 'acoustic guitar']
    
# Grab the corpus data
# FI = 1, IF = 2, RAND = ~3
input_feed, output_feed, generate_corpus, vocab = vp.generateCorpus(corpus, 1000, 1)

print(vocab)
