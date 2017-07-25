import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



def train(label, data):
    
    clf = svm.SVC(kernel='rbf') #0.526

    # clf = RandomForestClassifier() #0.45

    # clf = GradientBoostingClassifier() #0.426

    # clf = AdaBoostClassifier()  #0.442
    
    # clf = KNeighborsClassifier() # 0.426

    #clf = LogisticRegression()
    clf.fit(data, label)
    return clf 


