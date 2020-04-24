import pandas as pd
import numpy as np
import numpy.matlib
import sys

# Initializations
# ====================================================================================
# Set k based on commandline arguments
k = int(sys.argv[1])
printTable = int(sys.argv[2])

dataset = pd.read_csv('LabeledUserScores.csv')

# Training data
train_data = dataset.iloc[:742, :]
Y_train = train_data.iloc[:, 0]
X_train = train_data.iloc[:, 2:]

# Test data
test_data = dataset.iloc[742:, :]
Y_groundtruth = test_data.iloc[:, 0]
X_test = test_data.iloc[:, 2:]
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


# ====================================================================================
# Set k and run the prediction
results = pd.DataFrame()
results['Groundtruth'] = Y_groundtruth
predictions = list()
match = list()

# Make predictions about the test data
for i in range(len(X_test)):
    predictions.append(predict_from_query(X_train, X_test.iloc[i], k, Y_train))
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
print(f'Accuracy: {(matchTrue/(matchTrue + matchFalse)*100)}%')
# ====================================================================================
