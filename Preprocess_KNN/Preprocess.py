import pandas as pd
import numpy as np
from sklearn import preprocessing


# Label redditors in to 9 different politcal categories
def CategorizeUsers():

    labels = pd.read_csv('masterList_v2.csv')

    # Cut out the /u/ in front of the names
    labels['name'] = labels['name'].apply(lambda x: x[3:])

    categories = {
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

    # Replace categories with integers
    for i in categories:
        labels['quadrant'] = labels['quadrant'].replace(i, categories[i])

    labels.to_csv('labels.csv')


def Normalize():
    scores = pd.read_csv('reddior_scores.csv')

    # Replace all NaN with zeroes
    scores = scores.fillna(0)

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


def CombineLabelsScores():
    # Need to match up labels to scores

    labels = pd.read_csv('labels.csv')
    scores = pd.read_csv('normalized_scores.csv')

    # Cut out unecessary columns
    # scores = scores.iloc[:, 1:]
    labels = labels.iloc[:, 2:]

    user_labels = pd.Series([])

    # Look through each username in label data and match up names in scores data
    for i in range(len(labels)):

        user = pd.DataFrame(scores.loc[scores['names'] == labels.iloc[i, 0]])
        if not user.empty:
            user_labels[user.iloc[0, 0]] = labels.iloc[i, 1]

    scores.insert(1, 'Label', user_labels)

    scores = scores.iloc[:, 1:]

    scores.to_csv('LabeledUserScores.csv', index=False)


CategorizeUsers()
Normalize()
CombineLabelsScores()
