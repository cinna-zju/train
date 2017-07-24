import numpy as np
import load_data as ld
import train as tr
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import sqlite3
from sklearn.externals import joblib
# 392, 522, 782, 912, 1172, 1562,  2732   
# 8
folder = [2, 132,  392, 522, 782, 
    912, 1172, 1562, 1692, 1952, 
    2082,2212, 2342, 2472, 2602, 
    2862, 2992, 3382, 3512, 3642, 
    3772] #14*20-3 = 
for i in folder:
    test_folder = []
    print("test_folder: ", i)
    test_folder.append(i)
    train_folder = folder[:]
    train_folder.remove(i)
    label, data, loss = ld.get_data(20, train_folder)
    label_test, data_test, loss_test = ld.get_data(1, test_folder)

    accu = []
    clf = tr.train(label, data)
    # confusion = np.zeros([3,3])
    cnt = 0
    for j in range(0, data_test.shape[0]):
        result = clf.predict(data_test[j,:].reshape(1, -1))
        if label_test[j] == result:
            cnt += 1
        #confusion[int(label[j]+1), int(result+1)] += 1
    #print(confusion/np.sum(confusion))    
    accu.append(cnt/data_test.shape[0])    
    print('accuracy: ', accu, 'train sample: ', label.shape[0], 'test sample: ', label_test.shape[0])
print('average: ', np.mean(accu))



# data, mat(:,36) 4 channel * 9 emotion
# data = alldata[:, 2:11]
# data = np.column_stack((data, alldata[:, 13:22]))
# data = np.column_stack((data, alldata[:, 24:33]))
# data = np.column_stack((data, alldata[:, 35:44]))

# print(data.shape)
#print(label[0:200])
#clf = tr.train(label, alldata)

# # error
# for i in range(0, 4):
#     intdata = np.array(data[:, 7 + i*9])
#     # print(intdata[intdata<0])
#     for j in range(0, intdata.shape[0]):
#         if intdata[j] > 30:
#             intdata[j] = 1
#         else:
#             if intdata[j] < 0:
#                 intdata[j] = -1
#             else:
#                 intdata[j] = 0
#     print(np.mean(intdata == label))

#     #plt.subplot(6,1,i+1)
#     #plt.scatter(time, intdata, s=2)
#     #plt.scatter(time, data[:, 8+9*i], s=2)
#     #plt.ylim([-1,1])

# res = []
# for i in range(0, data.shape[0]):
# # for i in range(0, 17):
#     res.append(clf.predict(data[i, :].reshape(1, -1)))
# #plt.subplot(6,1,5)
# #plt.plot(time, res)

# #plt.subplot(6,1,6)
# #plt.plot(time, label)
#print(np.mean(res == label))
# #plt.show()

# for i in range(0, 4):
#     plt.subplot(5,1,i+1)
#     plt.plot(alldata[0:17,0], alldata[0:17,2+11*i],
#         alldata[0:17,0], alldata[0:17,3+11*i],
#         alldata[0:17,0], alldata[0:17,4+11*i],
#         alldata[0:17,0], alldata[0:17,5+11*i],
#         alldata[0:17,0], alldata[0:17,6+11*i],
#         alldata[0:17,0], alldata[0:17,7+11*i],
#         alldata[0:17,0], alldata[0:17,8+11*i]
#     )

# plt.subplot(5,1,5)
# plt.plot(alldata[0:17, 0], res[:17])
# plt.show()

# conn = sqlite3.connect('./data/2')
# c = conn.cursor()
# subdata = []
# for no in range(1,5):
#     sql = 'select * from emotions30_'+str(no)
#     c.execute(sql)
#     subdata.append(c.fetchall())
#     subdata[no-1] = np.array(subdata[no-1], dtype=np.float32)

# size = min(subdata[0].shape[0],
#     subdata[1].shape[0],
#     subdata[2].shape[0],
#     subdata[3].shape[0]
# )
# temp = subdata[0][0:size, :]
# for j in range(1, 4):
#     temp = np.column_stack((temp, subdata[j][0:size, :]))

# for j in range(0,4):
#     plt.subplot(6,1,j+1)
#     # print(temp[:, 9+11*j])
#     plt.plot(temp[:,0], temp[:, 9+11*j]/100)
#     plt.ylim([-1.5,1.5])

# EMOMASK = [0,0,1,1,1,1,1,1,1,1,1,
#     0,0,1,1,1,1,1,1,1,1,1,
#     0,0,1,1,1,1,1,1,1,1,1,
#     0,0,1,1,1,1,1,1,1,1,1
# ]

# EMOMASK = np.array(EMOMASK, dtype=np.bool)
# test = temp[:, EMOMASK]
# res = []
# for j in range(0, test.shape[0]):
#     res.append(clf.predict(test[j,:].reshape(1, -1)))

# plt.subplot(6,1,5)
# plt.plot(temp[:,0], res)

# plt.ylim([-1.5,1.5])

# plt.subplot(6,1,6)
# sublabel = 0.4 * test[:,7] +0.25 * test[:,16] + 0.25 * test[:,25] + 0.1 * test[:,34]
# l = 0
# while l < sublabel.shape[0]:
#     if sublabel[l] > 10:
#         sublabel[l] = 1
#     else:
#         if sublabel[l] < 0:
#             sublabel[l] = -1
#         else:
#             sublabel[l] = 0
#     l += 1 
# plt.plot(temp[:,0], sublabel)
# plt.ylim([-1.5, 1.5])
# plt.show()