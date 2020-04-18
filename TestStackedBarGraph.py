import csv
import matplotlib.pyplot as plt
import pandas as pd

# read CSV of saved redditor scores
redditor_scores = pd.read_csv('reddior_scores.csv', header=0)

# print(redditor_scores.loc[0])

columnList = (redditor_scores.columns.tolist())[1:]
scores = redditor_scores.iloc[1,1:].values.tolist()
normScores = [float(i) / sum(scores) for i in scores]
print(normScores)

plt.bar(columnList, normScores)
plt.show()