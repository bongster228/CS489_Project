import pandas as pd

labels = pd.read_csv('name_quadrant.csv')

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
