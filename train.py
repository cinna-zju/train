# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import tensorflow as tf
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib



def train(label, data):
    
    print('negetive:', label[label<0].shape[0])
    print('neutral:', label[label==0].shape[0])
    print('positive:', label[label>0].shape[0])

    # clf = svm.SVC()
    # clf.fit(data, label)


    rf = RandomForestClassifier()
    rf.fit(data, label)
    return rf

    return clf