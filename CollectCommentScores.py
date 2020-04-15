import praw
import comments
import time
import pandas as pd
import re
import csv

# Read CSV of redditors with scores calculated by PolCompBot
masterList = pd.read_csv('masterList.csv',
                        header=0,
                        names=['name','libAuth','lefRit','quadrant'])

# Create Reddit instance
reddit = praw.Reddit(client_id='a8fxlGxtt5HeRg', client_secret='x0oQ53axICf_azi5SY_yB5xVkE8', user_agent='Reddit Scrape')

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
        start = time.time() # start time
        try:
            # Get user comment karma
            temp = comments.get_comment_score_per_sub(reddit, redditor.name)
        except:
            print("Missing redditor!")
            temp = {}
        print("Scores retrieved for ", redditor.name)
        print("Time elapsed: ", (time.time() - start)) # end time
    if temp != {}:
        # Add retrieved scores to database
        redditor_scores = redditor_scores.append(temp, ignore_index=True)

# print(redditor_scores)

# Save scores to CSV
redditor_scores.to_csv('reddior_scores.csv', index=False)