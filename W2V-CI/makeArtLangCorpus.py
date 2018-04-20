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
           'woman break truck',
           'man stop car',
           'woman stop car',
           'man stop truck',
           'woman stop truck']

dishList = ['man break plate',
            'woman break plate',
            'man break glass',
            'woman break glass',
            'man smash plate',
            'woman smash plate',
            'man smash plate',
            'woman smash plate']



# multiply the sentences
# (keep them seperated if we want to do something later)
numS = 1000
vehCorpus = []
dishCorpus = []

for sentence in vehList:
    vehCorpus.append([sentence]*numS)
        
for sentence in dishList:
    dishCorpus.append([sentence]*numS)

# make list of all random sentences
vehFlat = [i for sL in vehCorpus for i in sL]
dishFlat = [i for sL in dishCorpus for i in sL]

# now combine them to make the corpus
artLang = vehFlat #+ dishFlat
artLangShuffled = random.sample(artLang, len(artLang))

print(len(artLangShuffled), 'total sentenses')
# save the shuffled data to a txt file
print('Writing artificial corpus...')
os.chdir('corpus')
t = 'artLang-' + str(len(artLangShuffled)) + '_' + str(numS)+ '-8s-1t_hom.txt'
print('Saved to', t)
with open(t, mode='wt', encoding='utf-8') as f:
        f.write('\n'.join(artLangShuffled))
