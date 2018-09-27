#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 13:19:47 2018

@author: willamannering
"""

import simmatrix_tensorflow_code

#genCoSim(numReps, fDom, aDom, numRuns, order)
#To use genCoSim:
    #numReps = number of repetitions
    #fDom = boolean, whether fish sense is dominant
    #aDom = boolean, whether acoustic sense is dominant
    #numRuns = number of times to run
    #order = which order to train data, values = 'random', 'fish first', or 'acoustic first'
    
    #returns array [bass-acoustic cosine similarity, bass-acoustic std, bass-trout cosine similarity, bass-trout std]

#%% No dominance, just order effects
    
#in this case, both cosine similarities should be about 50%
noDomRandOrder=genCoSim(1000, False, False, 20, 'random')


#in this case, the bass-acoustic cosim should be higher because the model was trained on acoustic last
noDomAcouLast=genCoSim(1000, False, False, 20, 'acoustic last')


#in this case, the bass-trout cosim should be higher because the model was trained on fish last
noDomFishLast=genCoSim(1000, False, False, 20, 'fish last')

#%% Random order, Acoustic dom

#in this case, the acoustic sense should have a higher cosim because it is the dominant sense and no order effects are present
randAcouDom = genCoSim(1000, False, True, 20, 'random')


#%%Random order, fish dom

#in this case, the fish sense should have a higher cosim because it is the dominant sense and no order effects are present
randFishDom = genCoSim(1000, True, False, 20, 'random')

#%%

#in this case, the fish sense cosim should be higher because fish was presented last even though the acoustic sense is dominant
acouDomFishLast = genCoSim(1000, False, True, 20, 'fish last')


#%%

#in this case, the acoustic sense cosim should be higher because acoustic was presented last even though the fish sense is dominant
fishDomAcouLast = genCoSim(1000, True, False, 20, 'acoustic last')