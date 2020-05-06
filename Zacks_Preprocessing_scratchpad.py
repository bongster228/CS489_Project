import pandas as pd
from scipy import stats 
import csv

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
"""
std_dev = redditor_scores['AskReddit'].std()
mean = redditor_scores['AskReddit'].mean()
z_col = (redditor_scores['AskReddit'] - mean) / std_dev
"""
print(total)

total.to_csv('sorted_znorm_scores.csv', index=False)