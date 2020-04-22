import pandas as pd
import numpy as np
from sklearn import preprocessing


scores = pd.read_csv('reddior_scores2.csv')

# Separate names and comment scores
names = scores.iloc[:, 0]
scores = scores.iloc[:, 1:]

# Normalize the scores
column_max = scores.max()
df_max = column_max.max()
normalized = scores / df_max

# Combine the scores and names
pd.DataFrame.insert(normalized, 0, "names", names)

pd.DataFrame.to_csv(normalized, 'normalized_scores.csv')
