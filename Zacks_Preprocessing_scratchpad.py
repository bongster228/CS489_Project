import pandas as pd
from scipy import stats
import numpy as np
import csv
import matplotlib.pyplot as plt

"""
# read CSV of saved redditor scores
redditor_scores = pd.read_csv('reddior_scores.csv', header=0,index_col='name')


total = redditor_scores.reindex(redditor_scores.sum().sort_values(ascending=False).index, axis=1)

print(total)

# Z-score subreddits
for col in (list(total.columns)):
    # total[col] = (total[col] - total[col].mean(skipna=True)/total[col].std(skipna=True))
    std_dev = total[col].std()
    mean = total[col].mean()
    total[col] = (total[col] - mean) / std_dev

# sums = redditor_scores.sum(axis=0, skipna=True).sort_values()
# sort columns by value

std_dev = redditor_scores['AskReddit'].std()
mean = redditor_scores['AskReddit'].mean()
z_col = (redditor_scores['AskReddit'] - mean) / std_dev

print(total)


total.to_csv('sorted_znorm_scores.csv', index=True)


# read CSV of saved redditor scores
redditor_scores = pd.read_csv('reddior_scores.csv', header=0,index_col='name')

count = redditor_scores['declutter'].count()

print(count)

i = 0

for col in list(redditor_scores.columns):
    if redditor_scores[col].count() < 2:
        i+=1
        print("Dropping col: " + col)
        redditor_scores.drop(columns=col, inplace=True)
        print(str(i) + " row deleted")

redditor_scores.to_csv('1_dropped_subs.csv', index=True)

print(str(i) + "cols removed")


for col in list(redditor_scores.columns):
    if redditor_scores[col].count() == 0:
        print("Dropping col: " + col)
        redditor_scores.drop(columns=col, inplace=True)

subreddit_counts = pd.DataFrame(redditor_scores.count().sort_values())
print(subreddit_counts)

redditor_scores = pd.read_csv('1_dropped_subs.csv', header=0,index_col='name')

subreddit_counts = pd.DataFrame(redditor_scores.count().sort_values())

boxplot = subreddit_counts.boxplot(column=0)
plt.show()

Q1 = subreddit_counts.quantile(.25)
Q3 = subreddit_counts.quantile(.75)
IQR = Q3-Q1
print(IQR)

out_thres = Q3 + (1.5 * IQR)
i = 0
for col in list(redditor_scores.columns):
    if redditor_scores[col].count() > int(out_thres):
        i+=1
        print("Dropping col: " + col + " " + str(i))
        redditor_scores.drop(columns=col, inplace=True)

redditor_scores.to_csv('Outliers_dropped.csv', index=True)

redditor_scores = pd.read_csv('Outliers_dropped.csv', header=0,index_col='name')

subreddit_counts = pd.DataFrame(redditor_scores.count().sort_values())

boxplot = subreddit_counts.boxplot(column=0)
plt.show()
"""
redditor_scores = pd.read_csv('Outliers_dropped.csv', header=0,index_col='name')

# Min-max subreddits
for col in (list(redditor_scores.columns)):
    smin = redditor_scores[col].min()
    smax = redditor_scores[col].max()
    smax_min = smax - smin
    redditor_scores[col] = (redditor_scores[col] - smin) / smax_min

redditor_scores.to_csv('Minmax_outliers_1dropped.csv', index=True)