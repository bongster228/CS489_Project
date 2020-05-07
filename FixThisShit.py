import praw
import comments
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore
import pprint
import operator
import re
import numpy as np
from matplotlib.patches import Rectangle

label_catergories = {
    "AuthLeft": 1,
    "LibLeft": 2,
    "AuthRight": 3,
    "LibRight": 4,
    "RightUnity": 5,
    "LeftUnity": 6,
    "LibUnity": 7,
    "AuthUnity": 8,
    "Centrist": 9
}

"""
label_list = [0,0,0,0,0,0,0,0,0]
n,p = masterList.shape
for label in masterList.quadrant:
    label_list[int(label_catergories[label]-1)]+=1

print(label_list)

plt.plot(label_list)
plt.show()

for i in range(0,9):
    label_list[i] = label_list[i]/n

print(label_list)
"""
redditor_scores = pd.read_csv('705_scores.csv', header=0)

masterList = pd.read_csv('balancedMaster.csv',
                        header=0,
                        names=['name','libAuth','lefRit','quadrant'])

print(redditor_scores.shape)

masterList = masterList[masterList.name.isin(redditor_scores['name'])]

print(masterList.shape)

masterList.to_csv('705_labels.csv')
