# corpus testing
import os
import numpy as np
import pandas as pd

# Grab the data
os.chdir("corpus")
print(os.getcwd())
data = open("biasArtLang_8xVeh-667_Dish-1333.txt").read().splitlines()
os.chdir("..")

senVeh = []
senDish = []
for sent in data:
    if sent.split()[2] == "car" or sent.split()[2] == "truck":
        senVeh.append(sent)
    elif sent.split()[2] == "glass" or sent.split()[2] == "plate":
        senDish.append(sent)

print('VehLen:', len(senVeh))
print('DishLen:', len(senDish))

