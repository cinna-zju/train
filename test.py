# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import tensorflow as tf
import numpy as np
import load_data as ld
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt


label, alldata, loss = ld.get_data(15)
print('negetive:', label[label<0].shape)
print('neutral:', label[label==0].shape)
print('positive:', label[label>0].shape)
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

# #     clf = svm.SVC(kernel = 'linear')
# #     clf.fit(data[train,:], label[train])
# #     cnt = 0
# #     for i in test:
# #         if label[i] == clf.predict(data[i, :].reshape(1, -1)):
# #             cnt += 1
# #     print('svm_linear accuracy: ', cnt / len(test))

#     # rf = RandomForestClassifier(n_jobs = -1)
#     # rf.fit(data[train, :], label[train])
#     # cnt = 0
#     # for i in test:
#     #     if label[i] == rf.predict(data[i, :].reshape(1, -1)):
#     #         cnt += 1
#     # print('rf accuracy: ', cnt / len(test))

time = range(0,label.shape[0])

# error
for i in range(0, 4):
    intdata = np.array(data[:, 7 + i*9])
    # print(intdata[intdata<0])
    for j in range(0, intdata.shape[0]):
        if intdata[j] > 30:
            intdata[j] = 1
        else:
            if intdata[j] < 0:
                intdata[j] = -1
            else:
                intdata[j] = 0
    print(np.mean(intdata == label))

    plt.subplot(6,1,i+1)
    plt.scatter(time, intdata, s=2)
    #plt.scatter(time, data[:, 8+9*i], s=2)
    plt.ylim([-1,1])

res = []
for i in range(0, data.shape[0]):
    res.append(clf.predict(data[i, :].reshape(1, -1)))
plt.subplot(6,1,5)
plt.plot(time, res)

plt.subplot(6,1,6)
plt.plot(time, label)
print(np.mean(res == label))
print('not found:', loss)
plt.show()


