from sklearn import svm
import numpy as np
x = np.array([[0,0], [1,1], [1,0]])

for i in x:
    print(i)
    print(i.shape)



# clf = svm.SVC()
# clf.fit(x,y)

# result = clf.predict([[2,2]])
# print(result)
# print(clf.support_vectors_)