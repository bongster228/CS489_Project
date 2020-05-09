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
    "Centrist": 7
}

quadrants = ['AuthLeft', 'LibLeft', 
            'AuthRight', 'LibRight',
            'RightUnity','LeftUnity', 'Centrist']

redditor_scores_template = pd.read_csv('scores_over_count_gt_30_pd_news.csv', header=0)
redditor_scores_template = redditor_scores_template.iloc[:150,:300]
redditor_scores = redditor_scores_template.sample(frac=1).reset_index()
del redditor_scores['index']
print(redditor_scores.shape)

masterList = pd.read_csv('masterListwoUnity_new.csv')
eval_out = open("results.txt", "w+")  

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

for v in range(0,3):
    total_predictions = 0
    total_predictions_correct = 0
    
    # Loop for K
    for k in range(7,13):
        # evaluation metrics variables
        name_count = {}
        quad_predictions = [0,0,0,0,0,0,0,0,0]
        cor_quad_predictions = [0,0,0,0,0,0,0,0,0]
        missed_quad_predictions = [0,0,0,0,0,0,0,0,0]
        ground_predictions = [0,0,0,0,0,0,0,0,0]
        predictions = 0
        predictions_correct = 0

        for z in range(0,10):
            # Shuffle data and labels
            # print("***SHUFFLING SAMPLES FOR " + str(k) + "***")
            redditor_scores = redditor_scores_template.sample(frac=.1).reset_index()
            del redditor_scores['index']
            
            h,w = redditor_scores.shape
            q = h- int(h//10)
            
            # Training data prep
            train_data = redditor_scores.iloc[:q, :]
            train_sub_count = [0,0,0,0,0,0,0,0,0]
            replace_array = {}

            # count quadrant frequency in training data
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
            
            # Bootstrapping ****
            # Check if quad is under-represented in training data
            # Duplicate random sample with sample label (from training data or from entire sample, if nec) 
            for i in range(0,7):
                while train_sub_count[i] <= int(h//9):
                    try:
                        train_data.append(replace_array[i].sample(n=1))
                    except:
                        train_data.append(redditor_scores.name[masterList.name[masterList.quadrant == quadrants[i]]].sample(n=1))
                    train_sub_count[i]+=1 
            
            # Shuffle bootstrapped training data
            train_data = train_data.sample(frac=1).reset_index()
            del train_data['index']
            
            # Apply Z-score normalization to training data
            names = list(train_data.columns)
            for col in names[1:]:
                std_dev = train_data[col].std()
                mean = train_data[col].mean()
                train_data[col] = (train_data[col] - mean) / std_dev
            
            # Split training data and index (for getting gt)
            Y_train = train_data.iloc[:, 0]
            X_train = train_data.iloc[:, 1:]
            
            # Test data prep
            test_data = redditor_scores.iloc[q:, :]
            # Z-score normalization w/ train data mean&std
            names = list(test_data.columns)
            for col in names[1:]:
                test_data[col] = (test_data[col] - mean) / std_dev
                
            # Get index for finding gt
            Y_groundtruth = test_data.iloc[:, 0]

            #Loop for tests
            print("***TRAINING FOR " + str(k) + "***")
            for j in range(0,test_data.shape[0]):
                # take row  of test data, convert to matrix matching training matrix
                X_test = test_data.iloc[j, 1:]
                X_test = np.matlib.repmat(X_test, int(X_train.shape[0]), 1).astype(float)
                
                # Calc distance
                # get difference between test and training data
                x_subtracted = np.subtract(X_test, X_train).fillna(0)
                
                # Square and sum that difference
                x_square = np.square(x_subtracted)

                # Sum the squared distance       
                x_sum = x_square.sum(axis = 1).astype(int)

                # Get indexes of top k closest
                x_sorted = np.argsort(x_sum)

                # Get label prediction
                label_list = [0,0,0,0,0,0,0,0,0,0]
                for i in range(1,k+1):
                    gtstr = Y_train[x_sorted[i]]
                    index = (masterList.name[masterList.name == gtstr].index[0])
                    label_list[label_catergories[masterList.quadrant[index]]-1]+=1
                    if gtstr not in name_count: name_count[gtstr]=1
                    else: name_count[gtstr]+=1
                prediction = (label_list.index(max(label_list)))
                quad_predictions[prediction]+=1 # Eval metric
                index = (masterList.name[masterList.name == Y_groundtruth.iloc[j]].index[0])
                
                # print("Test Results*****\nGround trurth for " + Y_groundtruth.iloc[j] + ": " + masterList.quadrant[index])
                # print(label_list)
                # print("Predicted " + quadrants[prediction])
                ground_predictions[label_catergories[masterList.quadrant[index]]-1]+=1

                # Check if prediction correct
                if((label_catergories[masterList.quadrant[index]]-1) == prediction): 
                    predictions_correct+=1
                    # print("Correct!")
                    cor_quad_predictions[prediction]+=1
                else: missed_quad_predictions[label_catergories[masterList.quadrant[index]]-1]+=1
                predictions+=1
            # END TEST LOOP

        # Save metrics for 10-fold k
        total_predictions+=predictions
        total_predictions_correct+=predictions_correct
        # name_count = {k: v for k, v in sorted(name_count.items(), key=lambda item: item[1])}
        # print(name_count)
        ks_array[k] = (predictions_correct/predictions)
        eval_out.writelines("K=" + str(k) +" 10-fold Accuracy: " + str(predictions_correct/predictions)+"\n")
        eval_out.writelines("Prediction distribution:")
        eval_out.writelines(str(quad_predictions)+"\n")
        eval_out.writelines("Ground Truth distribution:")
        eval_out.writelines(str(ground_predictions)+"\n")
        eval_out.writelines("Correct prediction distribution:")
        eval_out.writelines(str(cor_quad_predictions)+"\n")
        eval_out.writelines("Missed prediction distribution:")
        eval_out.writelines(str(missed_quad_predictions)+"\n")
        eval_out.flush()
        # END FOLD LOOP

    """
    p1 = plt.bar(ind, cor_quad_predictions, color='r')
    p2 = plt.bar(ind, missed_quad_predictions, bottom = cor_quad_predictions,  color='b')
    plt.legend((p1[0], p2[0]), ('Correct', 'Missed'))

    plt.show() 
    # exit()

""" 
    plt.plot(ks_array[7:13])       
    print(str(total_predictions_correct/total_predictions))
    # END K LOOP

eval_out.close() 
plt.show()