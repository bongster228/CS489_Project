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

masterList = pd.read_csv('label_list_1_with_old.csv',
                        header=0,
                        names=['name','libAuth','lefRit','quadrant'])
print(masterList.shape)
"""
label_list = [0,0,0,0,0,0,0,0,0]
n,p = masterList.shape
for label in masterList.quadrant:
    print(label)
    label_list[int(label_catergories[label]-1)]+=1

print(label_list)

plt.plot(label_list)
plt.show()
"""
# Create Reddit instance
reddit = praw.Reddit(client_id='a8fxlGxtt5HeRg', client_secret='x0oQ53axICf_azi5SY_yB5xVkE8', user_agent='Reddit Scrape')

# Create instance of redditor for PolCompBot
redditor = reddit.redditor('PolCompBot')

# read CSV of saved redditor scores
redditor_scores_template = pd.read_csv('redditor_scores_1.csv', header=0)
redditor_scores = redditor_scores_template

# read CSV of saved redditor scores
redditor_counts_template = pd.read_csv('redditor_counts_1.csv', header=0)
redditor_counts = redditor_counts_template

# Get users who have PolCompBot score but missing comment karma
missing_users = masterList[~masterList['name'].isin(redditor_scores['name'])]['name']

i = 1

# Loop over users
for user in missing_users:
    redditor = reddit.redditor(re.sub("/u/",'',user))
    print(redditor.name)
    temp = {} # temporary dictionary to store scores
    temp2 = {} # temporary dictionary to store scores
    if redditor.name in redditor_scores['name'].values:
        print(redditor.name, "is already saved")
    elif redditor.name not in redditor_scores['name'].values:
        # start = time.time() # start time
        try:
            # Get user comment karma
            temp, temp2 = comments.get_comment_score_per_sub(reddit, redditor.name)
        except:
            print("Missing redditor!")
            temp = {}
        # print("Scores retrieved for ", redditor.name)
        # print("Time elapsed: ", (time.time() - start)) # end time
    if temp != {}:
        # Add retrieved scores to database
        redditor_scores = redditor_scores.append(temp, ignore_index=True)
        redditor_counts = redditor_counts.append(temp2, ignore_index=True)
        i+=1
        print("Added " + str(redditor.name))

    if ((i%50) == 0) or (i==5):
        print(str(i) + " SAVING ")
        redditor_scores.to_csv('reddior_scores_' + str(i)+'.csv', index=False)
        redditor_counts.to_csv('reddior_counts_' + str(i)+'.csv', index=False)
        redditor_counts = redditor_counts_template
        redditor_scores = redditor_scores_template

redditor_scores.to_csv('reddior_scores' + str(i)+'.csv', index=False)
redditor_counts.to_csv('reddior_counts' + str(i)+'.csv', index=False)