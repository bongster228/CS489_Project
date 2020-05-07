import numpy as np
import pandas as pd
from numpy import random
import numpy.matlib

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
redditor_scores = pd.read_csv('705_scores.csv', header=0)
masterList = pd.read_csv('705_labels.csv')


"""
names = redditor_scores['name']
print(names)


# Shuffle data and labels
np.random.shuffle(redditor_scores.to_numpy())
redditor_scores = pd.DataFrame(redditor_scores)
"""
ks_array = [0]*23

n,p = redditor_scores.shape
print(redditor_scores.shape)

print(masterList.shape)
for k in range(9,10):
    # Shuffle data and labels
    np.random.shuffle(redditor_scores.to_numpy())
    redditor_scores = pd.DataFrame(redditor_scores)
    # Training data
    train_data = redditor_scores.iloc[:634, :]
    Y_train = train_data.iloc[:, 0]
    X_train = train_data.iloc[:, 1:]

    # Test data
    test_data = redditor_scores.iloc[634:, :]
    Y_groundtruth = test_data.iloc[:, 0]

    predictions = 0
    predictions_correct = 0
    
    #Loop for tests
    for j in range(0,test_data.shape[0]):
        X_test = test_data.iloc[j, 1:]

        # take row  of test data, convert to matrix matching training matrix
        X_test = np.matlib.repmat(X_test, int(X_train.shape[0]), 1).astype(float)

        # Find Euclidean distance
        # get difference between test and training data
        x_subtracted = np.subtract(X_test, X_train)

        # Square that difference
        x_square = np.square(x_subtracted)

        # Sum the squared distance       
        x_sum = x_square.sum(axis = 1).astype(int)

        # Get indexes of top k closest
        x_sorted = np.argsort(x_sum)

        label_list = [0,0,0,0,0,0,0,0,0,0]

        for i in range(1,k+1):
            gtstr = Y_train[x_sorted[i]]
            index = (masterList.name[masterList.name == gtstr].index[0])
            label_list[label_catergories[masterList.quadrant[index]]]+=1

        print(label_list)

        prediction = (label_list.index(max(label_list)))

        index = (masterList.name[masterList.name == Y_groundtruth.iloc[j]].index[0])
        if(label_catergories[masterList.quadrant[index]] == prediction): 
            predictions_correct+=1
            print("Correct!")

        predictions+=1
    ks_array[k] = (predictions_correct/predictions)
    print(predictions_correct/predictions)