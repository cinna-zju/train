import numpy as np
import load_data as ld
import train as tr
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import sqlite3

label, alldata, loss = ld.get_data(15)

print(alldata.shape)
print(label.shape)

# data, mat(:,36) 4 channel * 9 emotion
data = alldata[:, 2:11]
data = np.column_stack((data, alldata[:, 13:22]))
data = np.column_stack((data, alldata[:, 24:33]))
data = np.column_stack((data, alldata[:, 35:44]))

print(data.shape)

clf = tr.train(label, data)

# time = range(0,label.shape[0])

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
# print(np.mean(res == label))
# print('not found:', loss)
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

conn = sqlite3.connect('./data/2')
c = conn.cursor
subdata = []
for no in range(1,5):
    sql = 'select * from emotions2_'+str(no)
    c.execute(sql)
    subdata.append(c.fetchall())
    subdata[no-1] = np.array(subdata[no-1], dtype=np.float32)

size = min(subdata[0].shape[0],
    subdata[1].shape[0],
    subdata[2].shape[0],
    subdata[3].shape[0]
)
temp = subdata[0][0:size, :]
for j in range(1, 4):
    temp = np.column_stack((temp, subdata[j][0:size, :]))

for j in range(0,4):
    plt.subplot(5,1,i+1)
    plt.plot(temp[:,0], temp[:, 2+11*j],
        temp[:,0], temp[:, 3+11*j],
        temp[:,0], temp[:, 4+11*j],
        temp[:,0], temp[:, 5+11*j],
        temp[:,0], temp[:, 6+11*j],
        temp[:,0], temp[:, 7+11*j],
        temp[:,0], temp[:, 8+11*j]
    )

EMOMASK = [0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,1,1,1,1,1,1,0,0
]
test = temp[:, EMOMASK]
res = []
for j in range(0, test.shape[0]):
    res.append(clf.predict(test[i,:].reshape(1, -1)))

plt.subplot(5,1,5)
plt.plot(temp[:,0], res)