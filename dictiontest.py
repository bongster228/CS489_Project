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
import csv
import numpy as np
from matplotlib.patches import Rectangle

# Read CSV of redditors with scores calculated by PolCompBot
masterList = pd.read_csv('masterList.csv',
                        header=0,
                        names=['name','libAuth','lefRit','quadrant'])

# Create Reddit instance
reddit = praw.Reddit(client_id='a8fxlGxtt5HeRg', client_secret='x0oQ53axICf_azi5SY_yB5xVkE8', user_agent='Reddit Scrape')

redditor_scores = pd.read_csv('reddior_scores.csv', header=0)

missing_users = masterList[~masterList['name'].isin(redditor_scores['name'])]['name']

for user in missing_users[0:550]:
    redditor = reddit.redditor(re.sub("/u/",'',user))
    temp = {}
    if redditor.name in redditor_scores['name'].values:
        print(redditor.name, " is already saved")
    elif redditor.name not in redditor_scores['name'].values:
        start = time.time()
        try:
            temp = comments.get_comment_score_per_sub(reddit, redditor.name)
        except:
            print("Missing redditor!")
            temp = {}
        print("Scores retrieved for ", redditor.name)
        print("Time elapsed: ", (time.time() - start))
    if temp != {}:
        redditor_scores = redditor_scores.append(temp, ignore_index=True)

# print(redditor_scores)

redditor_scores.to_csv('reddior_scores.csv', index=False)
