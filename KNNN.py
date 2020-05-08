import numpy as np
import pandas as pd
from numpy import random
import numpy.matlib
import matplotlib.pyplot as plt

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

quadrants = ['AuthLeft', 'LibLeft', 
            'AuthRight', 'LibRight',
            'RightUnity','LeftUnity',
            'LibUnity', 'AuthUnity', 'Centrist']

redditor_scores_template = pd.read_csv('reddior_counts_combined_trimmed1.csv', header=0)
redditor_scores_template = redditor_scores_template.iloc[0:300,:400]
redditor_scores = redditor_scores_template
masterList = pd.read_csv('Labels_with_scores.csv')


"""
names = redditor_scores['name']
print(names)


# Shuffle data and labels
np.random.shuffle(redditor_scores.to_numpy())
redditor_scores = pd.DataFrame(redditor_scores)
"""
ks_array = [0]*25

n,p = redditor_scores.shape
print(redditor_scores.shape)

ind = np.arange(9)
width = 0.35 

print(masterList.shape)
total_predictions = 0
total_predictions_correct = 0
for v in range(0,5):
    for k in range(8,9):
        name_count = {}
        quad_predictions = [0,0,0,0,0,0,0,0,0]
        cor_quad_predictions = [0,0,0,0,0,0,0,0,0]
        missed_quad_predictions = [0,0,0,0,0,0,0,0,0]
        predictions = 0
        predictions_correct = 0
        for z in range(0,10):
            # Shuffle data and labels
            # print("***SHUFFLING SAMPLES FOR " + str(k) + "***")
            redditor_scores = redditor_scores_template.sample(frac=.1).reset_index()
            del redditor_scores['index']
            
            h,w = redditor_scores.shape
            q = h- int(h//10)
            # Training data
            train_data = redditor_scores.iloc[:q, :]
            
            # print(masterList.quadrant[masterList.name[masterList.name == redditor_scores.name].index[0]].value_counts())
            # exit()
            # print(h//9)

            train_sub_count = [0,0,0,0,0,0,0,0,0]
            replace_array = {}

            for name in train_data.name:
                
                train_label = label_catergories[masterList.quadrant[masterList.name[masterList.name == name].index[0]]]-1
                """
                if train_sub_count[train_label] > int(h//9):
                    train_data = train_data.drop(index = train_data.name[train_data.name == name].index[0])
                else:
                """
                if not train_label in replace_array:
                    replace_array[train_label] = train_data.name[train_data.name == name]
                else:
                    replace_array[train_label].append(train_data.name[train_data.name == name], ignore_index=True)
                    
                train_sub_count[train_label]+=1
            
            # print(train_sub_count)
            
            for i in range(0,9):
                while train_sub_count[i] <= int(h//9):
                    try:
                        train_data.append(replace_array[i].sample(n=1))
                    except:
                        train_data.append(redditor_scores.name[masterList.name[masterList.quadrant == quadrants[i]]].sample(n=1))
                    train_sub_count[i]+=1 
            
            # print(train_sub_count)
            
            train_data = train_data.sample(frac=1).reset_index()
            del train_data['index']
            names = list(train_data.columns)
            for col in names[1:]:
                # total[col] = (total[col] - total[col].mean(skipna=True)/total[col].std(skipna=True))
                std_dev = train_data[col].std()
                mean = train_data[col].mean()
                train_data[col] = (train_data[col] - mean) / std_dev
            
            # print(train_data)
            # exit()
            Y_train = train_data.iloc[:, 0]
            X_train = train_data.iloc[:, 1:]
            
            # Test data
            test_data = redditor_scores.iloc[q:, :]
            
            Y_groundtruth = test_data.iloc[:, 0]
            # print(test_data)
            # exit()
            #Loop for tests
            # print("***TRAINING FOR " + str(k) + "***")
            
            for j in range(0,test_data.shape[0]):
                X_test = test_data.iloc[j, 1:]
                # take row  of test data, convert to matrix matching training matrix
                X_test = np.matlib.repmat(X_test, int(X_train.shape[0]), 1).astype(float)
                

                # Find Euclidean distance
                # get difference between test and training data
                x_subtracted = np.subtract(X_test, X_train)
                # print(x_subtracted)
                # Square that difference
                x_square = np.square(x_subtracted)

                # Sum the squared distance       
                x_sum = x_square.sum(axis = 1).astype(int)

                # Get indexes of top k closest
                x_sorted = np.argsort(x_sum)

                label_list = [0,0,0,0,0,0,0,0,0,0]

                for i in range(1,k+1):
                    gtstr = Y_train[x_sorted[i]]
                    # print(gtstr)
                    index = (masterList.name[masterList.name == gtstr].index[0])
                    # print(gtstr + "" + masterList.quadrant[index])
                    label_list[label_catergories[masterList.quadrant[index]]-1]+=1
                    if gtstr not in name_count: name_count[gtstr]=1
                    else: name_count[gtstr]+=1

                prediction = (label_list.index(max(label_list)))

                index = (masterList.name[masterList.name == Y_groundtruth.iloc[j]].index[0])
                # print("Test Results*****")
                # print("Ground trurth for " + Y_groundtruth.iloc[j] + ": " + masterList.quadrant[index])
                # print(label_list)
                # print("Predicted " + quadrants[prediction])
                quad_predictions[prediction]+=1

                if((label_catergories[masterList.quadrant[index]]-1) == prediction): 
                    predictions_correct+=1
                    # print("Correct!")
                    cor_quad_predictions[prediction]+=1
                else: missed_quad_predictions[label_catergories[masterList.quadrant[index]]-1]+=1
                predictions+=1

            total_predictions+=predictions
            total_predictions_correct+=predictions_correct
        name_count = {k: v for k, v in sorted(name_count.items(), key=lambda item: item[1])}
        # print(name_count)
        ks_array[k] = (predictions_correct/predictions)
        
        print("K " + str(k) +" Accuracy: " + str(predictions_correct/predictions))
        print("Prediction distribution:")
        print(quad_predictions)
        print("Correct prediction distribution:")
        print(cor_quad_predictions)
        print("Missed prediction distribution:")
        print(missed_quad_predictions)
    """
    p1 = plt.bar(ind, cor_quad_predictions, color='r')
    p2 = plt.bar(ind, missed_quad_predictions, bottom = cor_quad_predictions,  color='b')
    plt.legend((p1[0], p2[0]), ('Correct', 'Missed'))

    plt.show() 
    # exit()

""" 
    plt.plot(ks_array[3:18])       
print(str(total_predictions_correct/total_predictions))


plt.show()