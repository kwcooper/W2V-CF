import pandas as pd
from pandas import DataFrame
import random



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


numS = 10

vehCorpus = []
dishCorpus = []

for sentence in vehList:
    vehCorpus.append([sentence]*numS)
        
