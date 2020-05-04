
import pandas as pd
import numpy as np
import numpy.matlib
import sys
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


# Initializations
# ====================================================================================
# Set k based on commandline arguments
k = int(sys.argv[1])
printTable = int(sys.argv[2])

dataset = pd.read_csv('LabeledUserScores.csv')


# ====================================================================================

# Functions
# ====================================================================================

# Return a distance matrix consisting of all the distances
# between the row and each row in the data


def euclidean_distance(data, row):
    # Create a matrix of the same size as data with given row
    row_matrix = np.matlib.repmat(row, len(data), 1)

    # Create distance matrix using euclidean formula
    distance_matrix = numpy.subtract(data, row_matrix)
    distance_matrix = distance_matrix ** 2
    distance_matrix = distance_matrix.sum(axis=1)
    distance_matrix = numpy.sqrt(distance_matrix)
    return distance_matrix


# Find k nearest neighbors and their classification
def k_nearest_neighbors(data, row, k, label_data):
    dist_matrix = euclidean_distance(data, row)
    # Combine classification and distance into a dataframe
    k_nearest = pd.DataFrame(dist_matrix, columns=['Distance'])
    k_nearest['Label'] = label_data
    # Remove the row with distance 0
    k_nearest = k_nearest.sort_values(by=['Distance'])
    k_nearest = k_nearest.iloc[1:]
    return k_nearest.iloc[:k]


# Return the prediction from the query value
def predict_from_query(data, query, k, train_label):
    k_nearest = k_nearest_neighbors(data, query, k, train_label)
    return k_nearest.Label.mode()[0]
# ====================================================================================


kf = KFold(n_splits=5, random_state=None, shuffle=True)

X = dataset.iloc[:, 2:]
y = dataset.iloc[:, 0]

k_accuracy = []


for _k in range(3, 19, 2):
    acurracy = []
    print(_k)
    for train, test in kf.split(dataset):
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        y_train = y.iloc[train]
        y_test = y.iloc[test]

        # ====================================================================================
        # Set k and run the prediction
        results = pd.DataFrame()
        results['Groundtruth'] = y_test
        predictions = list()
        match = list()

        # Make predictions about the test data
        for i in range(len(X_test)):
            predictions.append(predict_from_query(
                X_train, X_test.iloc[i], _k, y_train))
            match.append(predictions[i] == results.iloc[i, 0])

        # Combine the results
        results['Predictions'] = predictions
        results['Match'] = match
        # Count the correct and wrong predictions
        matchTrue = results['Match'].value_counts()[1]
        matchFalse = results['Match'].value_counts()[0]
        # Display the results
        if(printTable):
            print(results)

        print(f'Correct: {matchTrue}  Incorrect: {matchFalse}')
        print(f'Accuracy: {(matchTrue/(matchTrue + matchFalse)*100):.2f}%')
        acurracy.append((matchTrue/(matchTrue + matchFalse)*100))
        # ====================================================================================`
    k_accuracy.append(acurracy.mean())

x_axis = [3, 5, 7, 9, 11, 13, 15, 17, 19]

plt.plot(x_axis, k_accuracy)
plt.xlabel('K value for KNN')
plt.ylabel('% Accuracy')
plt.title('Finding Optimal K')
plt.savefig('FindK.png')

print(k_accuracy)
# plt.plot(acurracy)
# plt.xlabel("Test Cases")
# plt.ylabel("Accuracy")
# plt.title("10 Fold Cross Validation")

# plt.savefig('graph.png')
