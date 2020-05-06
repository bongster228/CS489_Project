import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from scipy.stats import zscore

# read CSV of saved redditor scores
redditor_scores = pd.read_csv('reddior_scores.csv', header=0)

names_list = redditor_scores['name'].tolist

sort_scores = redditor_scores.sort_values(by=names_list,axis=1, ascending=False)

sort_scores.to_csv('sort_scores.csv', index=False)

