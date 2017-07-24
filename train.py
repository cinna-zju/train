import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib



def train(label, data):
    clf = svm.SVC(kernel='linear')
    clf.fit(data, label)
    return clf

    # rf = RandomForestClassifier()
    # rf.fit(data, label)
    # return rf

