import numpy as np
import pandas as pd
from numpy import random
import numpy.matlib
import matplotlib.pyplot as plt
"""
T1 = [0.13333333333333333,.16666666666666666,.23333333333333334,.23333333333333334,.23333333333333334,.2,.13333333333333333,0.2,0.13333333333333333,0.26666666666666666,0.13333333333333333,0.2,0.3333333333333333,0.3333333333333333,0.3],,,,
T2 = [.1,.16666666666666666,.23333333333333334,.2,.36666666666666664,.2,.06666666666666667,0.23333333333333334,0.1,0.36666666666666664,0.2,0.1,0.3,0.1,0.1],,,,
T3 = [.1,.2,.16666666666666666,.1,.16666666666666666,.2,.06666666666666667,0.2,0.23333333333333334,0.13333333333333333,0.4666666666666667,0.36666666666666664,0.13333333333333333,0.26666666666666666,0.23333333333333334],,,,
T4 = [.13333333333333333,.1,.1,.4,.2,.13333333333333333,.2,0.16666666666666666,0.3,0.26666666666666666,0.4,0.2,0.16666666666666666,0.23333333333333334,0.3],,,,
T5 = [.1,.36666666666666664,.06666666666666667,.2,.16666666666666666,.03333333333333333,.1,0.2,0.16666666666666666,0.2,0.06666666666666667,0.16666666666666666,0.2,0.23333333333333334,0.13333333333333333],,,,
T6 = [.2,.03333333333333333,.16666666666666666,.1,.06666666666666667,.3333333333333333,.2,0.3,0.2,0.16666666666666666,0.26666666666666666,0.36666666666666664,0.23333333333333334,0.13333333333333333,0.26666666666666666],,,,

Trials = [T1,T2,T3,T4,T5],,,,
for t in Trials:
    plt.plot(t)
labels = ('3','5','7','9','11','13','15','17')
ticks = plt.xticks()[0],,,,
plt.xticks(ticks,labels=('0','3','5','7','9','11','13','15','17','19'))
plt.xlabel("K = ")
plt.ylabel("Accuracy")
plt.show()
"""

# <array type>_<k>_<type indicator>_<trial>
data = {
    'PD_7_1':[5, 18, 5, 13, 0, 7, 2],
    'GT_7_2':[7, 7, 9, 8, 1, 10, 8],
    'TP_7_3':[2, 1, 0, 2, 0, 2, 0],
    'MP_7_4':[5, 6, 9, 6, 1, 8, 8],

    'PD_8_1':[0, 20, 18, 5, 0, 6, 1],
    'GT_8_2':[6, 11, 3, 7, 0, 17, 6],
    'TP_8_3':[0, 5, 0, 1, 0, 2, 0],
    'MP_8_4':[6, 6, 3, 6, 0, 15, 6],

    'PD_9_1':[1, 7, 8, 1, 0, 31, 2],
    'GT_9_2':[8, 7, 6, 3, 6, 12, 8],
    'TP_9_3':[0, 0, 2, 0, 0, 7, 0],
    'MP_9_4':[8, 7, 4, 3, 6, 5, 8],

    'PD_10_1':[6, 10, 12, 1, 0, 16, 5],
    'GT_10_2':[1, 6, 7, 5, 4, 15, 12],
    'TP_10_3':[1, 2, 1, 0, 0, 4, 1],
    'MP_10_4':[0, 4, 6, 5, 4, 11, 11],

    'PD_11_1':[0, 11, 6, 7, 0, 21, 5],
    'GT_11_2':[3, 9, 10, 7, 5, 15, 1],
    'TP_11_3':[0, 2, 1, 0, 0, 8, 0],
    'MP_11_4':[3, 7, 9, 7, 5, 7, 1],

    'PD_12_1':[5, 8, 15, 9, 0, 12, 1],
    'GT_12_2':[2, 12, 7, 3, 1, 18, 7],
    'TP_12_3':[0, 1, 4, 0, 0, 3, 0],
    'MP_12_4':[2, 11, 3, 3, 1, 15, 7],

    'PD_7_1':[9, 12, 17, 6, 0, 6, 0],
    'GT_7_2':[4, 5, 10, 8, 6, 14, 3],
    'TP_7_3':[1, 1, 2, 2, 0, 2, 0],
    'MP_7_4':[3, 4, 8, 6, 6, 12, 3],

    'PD_8_1':[1, 17, 6, 7, 0, 19, 0],
    'GT_8_2':[2, 7, 6, 7, 6, 19, 3],
    'TP_8_3':[0, 2, 0, 0, 0, 4, 0],
    'MP_8_4':[2, 5, 6, 7, 6, 15, 3],

    'PD_9_1':[3, 7, 5, 7, 1, 27, 0],
    'GT_9_2':[7, 9, 9, 3, 1, 12, 9],
    'TP_9_3':[1, 0, 0, 0, 0, 7, 0],
    'MP_9_4':[6, 9, 9, 3, 1, 5, 9],

    'PD_10_1':[10, 7, 13, 1, 1, 17, 1],
    'GT_10_2':[4, 8, 8, 11, 3, 12, 4],
    'TP_10_3':[0, 0, 1, 1, 0, 2, 0],
    'MP_10_4':[4, 8, 7, 10, 3, 10, 4],

    'PD_11_1':[2, 7, 4, 8, 0, 28, 1],
    'GT_11_2':[5, 8, 11, 7, 2, 13, 4],
    'TP_11_3':[0, 1, 1, 1, 0, 8, 0],
    'MP_11_4':[5, 7, 10, 6, 2, 5, 4],

    'PD_12_1':[1, 10, 12, 0, 0, 25, 2],
    'GT_12_2':[2, 16, 4, 10, 1, 13, 4],
    'TP_12_3':[0, 2, 0, 0, 0, 7, 0],
    'MP_12_4':[2, 14, 4, 10, 1, 6, 4],

    'PD_7_1':[7, 13, 12, 1, 0, 17, 0],
    'GT_7_2':[2, 5, 10, 9, 4, 13, 7],
    'TP_7_3':[0, 0, 3, 1, 0, 3, 0],
    'MP_7_4':[2, 5, 7, 8, 4, 10, 7],

    'PD_8_1':[6, 11, 12, 0, 0, 21, 0],
    'GT_8_2':[2, 12, 8, 8, 2, 14, 4],
    'TP_8_3':[0, 2, 1, 0, 0, 6, 0],
    'MP_8_4':[2, 10, 7, 8, 2, 8, 4],

    'PD_9_1':[8, 4, 11, 6, 0, 20, 1],
    'GT_9_2':[4, 6, 8, 6, 5, 13, 8],
    'TP_9_3':[1, 1, 2, 0, 0, 8, 0],
    'MP_9_4':[3, 5, 6, 6, 5, 5, 8],

    'PD_10_1':[4, 11, 12, 5, 5, 13, 0],
    'GT_10_2':[6, 5, 6, 7, 3, 15, 8],
    'TP_10_3':[0, 2, 1, 1, 0, 5, 0],
    'MP_10_4':[6, 3, 5, 6, 3, 10, 8],

    'PD_11_1':[7, 8, 15, 4, 0, 9, 7],
    'GT_11_2':[7, 7, 8, 8, 4, 12, 4],
    'TP_11_3':[0, 1, 3, 0, 0, 1, 1],
    'MP_11_4':[7, 6, 5, 8, 4, 11, 3],

    'PD_12_1':[6, 5, 11, 2, 0, 24, 2],
    'GT_12_2':[3, 10, 10, 5, 1, 12, 9],
    'TP_12_3':[0, 1, 3, 0, 0, 5, 0],
    'MP_12_4':[3, 9, 7, 5, 1, 7, 9], 
}

temp = {
    'PD_3_1':[7, 15, 5, 0, 0, 3, 0],
    'GT_3_2':[2, 4, 2, 3, 2, 14, 0, 1, 2],
    'TP_3_3':[0, 3, 1, 0, 0, 2, 0],
    'MP_3_4':[2, 1, 1, 3, 2, 12, 0, 1, 2],
}

TPR = [0,0,0,0,0,0,0,0,0]
i = 0
for j,k in zip(temp['TP_3_3'],temp['GT_3_2']):
    try: TPR[i] = j/k
    except: TPR[i] = 0
    i+=1

# plt.plot(TPR)
# plt.show()
print(TPR)

TTPR = [0,0,0,0,0,0,0]
TFPR = [0,0,0,0,0,0,0]

for i in range(7,12):
    j = 0
    
    TPR = [0,0,0,0,0,0,0]
    FPR = [0,0,0,0,0,0,0]

    cor_pre = data[str('TP_'+str(i)+"_3")]
    gts = data[str('GT_'+str(i)+"_2")]
    mis_pre = data[str('MP_'+str(i)+"_4")]
    tot_pred = data[str('PD_'+str(i)+"_1")]
    for q,r,s,t in zip(cor_pre,gts,mis_pre,tot_pred):
        try: TPR[j] = q/r
        except: TPR[j] = 0
        try: FPR[j] = s/r
        except: FPR[j] = 0
        print(FPR[j])
        TTPR[j] += float(TPR[j])
        TFPR[j] += float(FPR[j])
        j+=1
    plt.plot(TPR, color='r',alpha=0.4)
    plt.plot(FPR, color='b',alpha=0.30)

for i in range(0,7):
    TTPR[i] = TTPR[i]/6
    TFPR[i] = TFPR[i]/6
print(TFPR)
print(TTPR)

quadrants = ['AuthLeft', 'LibLeft', 
            'AuthRight', 'LibRight',
            'RightUnity','LeftUnity','Centrist']

plt.plot(TTPR, color='r',)
plt.plot(TFPR, color='b')
ticks = [0,1,2,3,4,5,6]
plt.xticks(ticks,labels=quadrants)
plt.legend(("TPR","FPR"))
plt.xlabel("Quadrant")
plt.ylabel("Accuracy")
plt.show()

"""

redditor_scores = pd.read_csv('scores_over_count_gt_30_pd.csv', header=0)

masterList = pd.read_csv('masterListwoUnity.csv',
                        header=0,
                        names=['name','libAuth','lefRit','quadrant'],)
masterList['name'], = masterList['name'],.apply(lambda x: x[3:],)

print(masterList.shape)
masterList = masterList[masterList.name.isin(redditor_scores['name'],)],
print(masterList.shape)

print(redditor_scores.shape)
redditor_scores = redditor_scores[redditor_scores.name.isin(masterList['name'],)],
print(redditor_scores.shape)

print(masterList.shape)
masterList = masterList[masterList.name.isin(redditor_scores['name'],)],
print(masterList.shape)

masterList.to_csv('masterListwoUnity_new.csv', index=True)
redditor_scores.to_csv('scores_over_count_gt_30_pd_news.csv', index=True)
"""
"""
redditor_scores = pd.read_csv('scores_over_count_gt_30_pd_news.csv', header=0)
print(redditor_scores.shape)
redditor_scores.],,duplicates(subset=['name'],, inplace=True)
print(redditor_scores.shape)
redditor_scores.to_csv('scores_over_count_gt_30_pd_news.csv', index=True)
"""