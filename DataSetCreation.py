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

# Create Data Base of redditors with scores calculated by PolCompBot
masterList = pd.read_csv('masterList.csv',
                        header=0,
                        names=['name','libAuth','lefRit','quadrant'])

row = (masterList.shape[0])+1
# Create Reddit instance
reddit = praw.Reddit(client_id='a8fxlGxtt5HeRg', client_secret='x0oQ53axICf_azi5SY_yB5xVkE8', user_agent='Reddit Scrape')

# Create instance of redditor for PolCompBot
redditor = reddit.redditor('PolCompBot')

for redditor_comment in redditor.comments.new(limit=None):
    body = redditor_comment.body # Get comment body
    
    # Get lib/auth left/right score from comment body, cast to float
    try:
        libAuthScore = float(re.sub(r'\*\*','',re.findall(r'(\*\*[\d,.,-]+\*\*)', body)[0]))
        lefRitScore = float(re.sub(r'\*\*','',re.findall(r'(\*\*[\d,.,-]+\*\*)', body)[1]))
    except:
        print("No Score")

    # check if both scores are 0
    if (libAuthScore == 0 and lefRitScore == 0) or not re.findall(r'(\*\*[\d,.,-]+\*\*)', body):
        print("No score")
    
    # else get user and quadrant
    else:
        user = re.search(r'/u/[^\s]+', body).group()
        quadrant = re.sub(r'\*\*','',re.search(r'(\*\*[\w]+\*\*)', body).group())
        tempDF = ({ "name":     user,
                    "libAuth":  [libAuthScore],
                    "lefRit":   [lefRitScore],
                    "quadrant": quadrant})
        
        # Check if user is already in masterlist
        if not (masterList['name'].str.contains(user).any()):
            masterList.loc[row] = tempDF
            row += 1

# Create Scatter Plot of scores
quadrants = ['LibRight', 'AuthUnity', 'LeftUnity', 'Centrist', 'RightUnity', 'LibLeft', 'AuthLeft', 'AuthRight', 'LibUnity']
quad_colors = ['#c19bed','#b493af', '#c0c080','#c0c0c0','#93afdc','#80ff80','#ff8080','#40acff','#aec5c3']

"""
fig = plt.figure()
for quad,col in zip(quadrants,quad_colors):
    cond_y = np.array(masterList[masterList['quadrant']==quad]['libAuth'].str.replace(r'[\[\]]','').astype(float))
    cond_x = np.array(masterList[masterList['quadrant']==quad]['lefRit'].str.replace(r'[\[\]]','').astype(float))
    plt.scatter(cond_x,cond_y,color=col)

plt.grid(True)
plt.xlabel('Left/Right')
plt.ylabel('Lib/Auth')
currentAxis = plt.gca()
# currentAxis.add_patch(Rectangle((-3.5, -3.5), 7, 7, fill = False)).set_edgecolor('#c0c0c0')

plt.show()
"""

# Save MasterList to CSV
masterList.to_csv("masterList.csv")

# read CSV of saved redditor scores
redditor_scores = pd.read_csv('reddior_scores.csv', header=0)

# Get users who have PolCompBot score but missing comment karma
missing_users = masterList[~masterList['name'].isin(redditor_scores['name'])]['name']

# Loop over users
for user in missing_users:
    redditor = reddit.redditor(re.sub("/u/",'',user))
    temp = {} # temporary dictionary to store scores
    if redditor.name in redditor_scores['name'].values:
        print(redditor.name, "is already saved")
    elif redditor.name not in redditor_scores['name'].values:
        # start = time.time() # start time
        try:
            # Get user comment karma
            temp = comments.get_comment_score_per_sub(reddit, redditor.name)
        except:
            print("Missing redditor!")
            temp = {}
        # print("Scores retrieved for ", redditor.name)
        # print("Time elapsed: ", (time.time() - start)) # end time
    if temp != {}:
        # Add retrieved scores to database
        redditor_scores = redditor_scores.append(temp, ignore_index=True)

# print(redditor_scores)

# Save scores to CSV
redditor_scores.to_csv('reddior_scores.csv', index=False)