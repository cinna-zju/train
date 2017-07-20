# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import tensorflow as tf
import numpy as np
import load_data as ld
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt

label, alldata = ld.get_data(8)
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
    confusion = np.zeros([3,3])
    clf = svm.SVC()
    clf.fit(data[train,:], label[train])
    cnt = 0
    for i in test:
        result = clf.predict(data[i, :].reshape(1, -1))
        if label[i] == result:
            cnt += 1
        confusion[int(label[i]+1), int(result+1)] += 1
    print('svm accuracy: ', cnt / len(test))
    print(confusion/np.sum(confusion))

#     clf = svm.SVC(kernel = 'linear')
#     clf.fit(data[train,:], label[train])
#     cnt = 0
#     for i in test:
#         if label[i] == clf.predict(data[i, :].reshape(1, -1)):
#             cnt += 1
#     print('svm_linear accuracy: ', cnt / len(test))

    # rf = RandomForestClassifier(n_jobs = -1)
    # rf.fit(data[train, :], label[train])
    # cnt = 0
    # for i in test:
    #     if label[i] == rf.predict(data[i, :].reshape(1, -1)):
    #         cnt += 1
    # print('rf accuracy: ', cnt / len(test))

time = range(0,label.shape[0])
# for i in range(1,5):
#     plt.subplot(5,1,i)
#     plt.plot(time, data[:,0+7*(i-1)],
#         time, data[:,1+7*(i-1)],
#         time, data[:,2+7*(i-1)],
#         time, data[:,3+7*(i-1)],
#         time, data[:,4+7*(i-1)],
#         time, data[:,5+7*(i-1)],
#         time, data[:,6+7*(i-1)],
#     )
plt.subplot(2,1,1)
plt.plot(time, label)

res = []
for i in range(0, data.shape[0]):
    res.append(clf.predict(data[i, :].reshape(1, -1)))
plt.subplot(2,1,2)
plt.plot(time, res)
plt.show()