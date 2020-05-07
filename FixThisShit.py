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
import os
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

redditor_scores = pd.read_csv('705_scores.csv', header=0)

masterList = pd.read_csv('balancedMaster.csv',
                        header=0,
                        names=['name','libAuth','lefRit','quadrant'])

print(redditor_scores.shape)

masterList = masterList[masterList.name.isin(redditor_scores['name'])]

print(masterList.shape)

masterList.to_csv('705_labels.csv')

# read CSV of saved redditor scores
redditor_scores_template = pd.read_csv('redditor_scores_1.csv', header=0)
redditor_scores = redditor_scores_template

# read CSV of saved redditor scores
redditor_counts_template = pd.read_csv('redditor_counts_1.csv', header=0)
redditor_counts = redditor_counts_template

list_of_files = {}

for (dirpath, dirnames, filenames) in os.walk(os.getcwd()):
    for filename in filenames:
        if re.search(r'(redditor|reddior)_scores(_|\d)\d', filename):
            print("Adding " + filename) 
            temp = pd.read_csv(filename, header=0)
            redditor_scores = redditor_scores.append(temp, ignore_index=True)
        if re.search(r'(redditor|reddior)_counts(_|\d)\d', filename): 
            print("Adding " + filename)
            temp2 = pd.read_csv(filename, header=0)
            redditor_counts = redditor_counts.append(temp2, ignore_index=True)

redditor_scores.to_csv('reddior_scores_combined.csv', index=False)
redditor_counts.to_csv('reddior_counts_combined.csv', index=False)

"""
temp = pd.read_csv('reddior_scores_combined.csv', header=0,index_col=0)
temp2 = pd.read_csv('reddior_counts_combined.csv', header=0,index_col=0)
temp = temp.astype(float, 'ignore')
temp2 = temp2.astype(float, 'ignore')
print(temp.shape)
print(temp2.shape)

temp3 = temp.divide(temp2)
print(temp2.shape)
temp3.to_csv('reddior_score_over_count.csv', index=True)