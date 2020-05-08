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


redditor_scores = pd.read_csv('Outliers_dropped.csv', header=0,index_col='name')

# Min-max subreddits
for col in (list(redditor_scores.columns)):
    smin = redditor_scores[col].min()
    smax = redditor_scores[col].max()
    smax_min = smax - smin
    redditor_scores[col] = (redditor_scores[col] - smin) / smax_min

redditor_scores.to_csv('Minmax_outliers_1dropped.csv', index=True)


# redditor_scores = pd.read_csv('reddior_score_over_count.csv', header=0)

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

# Create Data Base of redditors with scores calculated by PolCompBot
masterList = pd.read_csv('Labels_with_scores.csv',
                        header=0,
                        names=['name','libAuth','lefRit','quadrant'])

# Create Scatter Plot of scores
quadrants = ['LibRight', 'AuthUnity', 'LeftUnity', 'Centrist', 'RightUnity', 'LibLeft', 'AuthLeft', 'AuthRight', 'LibUnity']
quad_colors = ['#c19bed','#b493af', '#c0c080','#c0c0c0','#93afdc','#80ff80','#ff8080','#40acff','#aec5c3']

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


total = pd.read_csv('scores_over_count_gt_30_zscoresub_dupsdrop.csv', header=0)
redditor_scores = total

print(total.shape)
previousname = total.name[0]
for i in range(1,total.shape[0]):
    if total.name[i] == total.name[i-1]:
        print("Duplicate!")
        redditor_scores = redditor_scores.drop(index=i)

print(redditor_scores.shape)
redditor_scores.to_csv('scores_over_count_gt_30_zscoresub_dupsdrop.csv', index=True)
"""
redditor_scores = pd.read_csv('scores_over_count_gt_30_minmax.csv', header=0, index_col='name')

subreds = ['accidentallycommunist','chapotraphouse2','communism','communism101','debateacommunist','debatecommunism','fullcommunism','latestagecapitalism','moretankiechapo','genzedong']

for subr in subreds:
    if subr in redditor_scores.columns:
        redditor_scores.drop(columns=subr, inplace=True)

redditor_scores.to_csv('scores_over_count_gt_30_poldropped.csv')