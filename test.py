# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import tensorflow as tf
import numpy as np
import load_data as ld
from sklearn import svm
import random

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

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(data[0:1500,:], label[0:1500])

cnt = 0
for k in range(500):
    i = random.randint(1500, 2700)
    result = (clf.predict(data[i,:].reshape(1,-1)))
    if result == label[i]:
        cnt += 1

print(cnt/500)

#print(clf.support_vectors_)

