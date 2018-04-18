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
numS1 = 1333
numS2 = 667
vehCorpus = []
dishCorpus = []

for sentence in vehList:
    vehCorpus.append([sentence]*numS1)
print('\nVeh:', numS1)
        
for sentence in dishList:
    dishCorpus.append([sentence]*numS2)
print('Dish:', numS2)

# make list of all random sentences
vehFlat = [i for sL in vehCorpus for i in sL]
dishFlat = [i for sL in dishCorpus for i in sL]

# now combine and shuffle them to make the corpus
artLang = vehFlat + dishFlat
artLangShuffled = random.sample(artLang, len(artLang))

print('total len:', len(artLangShuffled))
# save the shuffled data to a txt file
print('\nWriting artificial corpus...')
os.chdir('corpus')
t = 'biasArtLang_8xVeh-' + str(numS1) + '_Dish-' + str(numS2) + '.txt'
print(t)
with open(t, mode='wt', encoding='utf-8') as f:
        f.write('\n'.join(artLangShuffled))


