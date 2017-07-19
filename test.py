# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import tensorflow as tf
import numpy as np
import load_data as ld
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import random

train_num = 2000
vali_num = 700

label, alldata = ld.get_data(6)
# emo_tag = allo_label(emo)
print(alldata.shape)

# data, mat(:,36) 4 channel * 9 emotion
data = alldata[:, 2:11]
data = np.column_stack((data, alldata[:, 13:22]))
data = np.column_stack((data, alldata[:, 24:33]))
data = np.column_stack((data, alldata[:, 35:44]))

print(data.shape)
print(label.shape)

skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(data, label):
        clf = svm.SVC(ker)
        clf.fit(data[train,:], label[train])
        cnt = 0
        for i in test:
            if label[i] == clf.predict(data[i, :].reshape(1, -1)):
                cnt += 1
        print('svm accuracy: ', cnt / len(test))

        rf = RandomForestClassifier()
        rf.fit(data[0:train_num, :], label[0:train_num])
        cnt = 0
        for i in test:
            if label[i] == rf.predict(data[i, :].reshape(1, -1)):
                cnt += 1
        print('rf accuracy: ', cnt / len(test))
