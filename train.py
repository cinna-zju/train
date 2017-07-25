import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier



def train(label, data):
    # clf = svm.SVC(kernel='sigmoid')
    # clf.fit(data, label)
    # return clf

    # rf = RandomForestClassifier()
    # rf.fit(data, label)
    # return rf

    # clf = GradientBoostingClassifier().fit(data, label)
    # return clf # 0.43

    #clf = AdaBoostClassifier().fit(data, label)  0.44
    
    clf = KNeighborsClassifier().fit(data, label) 0.42
    return clf 


