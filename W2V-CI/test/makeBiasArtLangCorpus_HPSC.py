# for hpsc

import pandas as pd
from pandas import DataFrame
import random
import os

d = [0,1,2,3]

df = DataFrame(data = d)

# Define the elman setences 
vehList = ['man break car',
           'woman break car',
           'man break truck',
           'woman break truck']

dishList = ['man break plate',
            'woman break plate',
            'man break glass',
            'woman break glass']

# multiply the setences
# (keep them seperated if we want to do something later)
numS1 = 500
numS2 = 1500
vehCorpus = []
dishCorpus = []

for sentence in vehList:
    vehCorpus.append([sentence]*numS1)
        
for sentence in dishList:
    dishCorpus.append([sentence]*numS2)

# make list of all random sentences
vehFlat = [i for sL in vehCorpus for i in sL]
dishFlat = [i for sL in dishCorpus for i in sL]

# now combine them to make the corpus
artLang = vehFlat + dishFlat
artLangShuffled = random.sample(artLang, len(artLang))

print('total len:', len(artLangShuffled))
# save the shuffled data to a txt file
print('Writing artificial corpus...')
os.chdir('corpus')
with open('artLang_2s_1-2Bias_v-d_8x1000_shuffled.txt', mode='wt', encoding='utf-8') as f:
        f.write('\n'.join(artLangShuffled))
