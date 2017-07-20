# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import tensorflow as tf
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib


def train(label, data):
    
    print('negetive:', label[label<0].shape)
    print('neutral:', label[label==0].shape)
    print('positive:', label[label>0].shape)


    # skf = StratifiedKFold(n_splits=3)
    # for train, test in skf.split(data, label):
    #     confusion = np.zeros([3,3])
    #     clf = svm.SVC()
    #     clf.fit(data[train,:], label[train])
    #     cnt = 0
    #     for i in test:
    #         result = clf.predict(data[i, :].reshape(1, -1))
    #         if label[i] == result:
    #             cnt += 1
    #         confusion[int(label[i]+1), int(result+1)] += 1
    #     print('svm accuracy: ', cnt / len(test))
    #     print(confusion/np.sum(confusion))



    clf = svm.SVC()
    clf.fit(data, label)
    joblib.dump(clf, 'dump.pkl')
    #clf = joblib.load('dump.pkl')
    cnt = 0


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




    return clf