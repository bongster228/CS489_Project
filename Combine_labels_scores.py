import pandas as pd

# Need to match up labels to scores

labels = pd.read_csv('labels.csv')
scores = pd.read_csv('normalized_scores.csv')

user_labels = pd.Series([])

# Look through each username in label data and match up names in scores data
for i in range(len(labels)):

    user = pd.DataFrame(scores.loc[scores['names'] == labels.iloc[i, 0]])
    # print(user)
    if not user.empty:
        user_labels[user.iloc[0, 0]] = labels.iloc[i, 1]

scores.insert(1, 'Label', user_labels)

scores.to_csv('LabeledUserScores.csv')
