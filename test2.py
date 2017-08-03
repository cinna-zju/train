import numpy as np
import load_data as ld
import train as tr

from matplotlib import pyplot as plt



# 392, 522, 782, 912, 1172, 1562,  2732   
# 8
# 1822 no siganl
folder = [2, 132,  262, 392, 522, 
652, 782,912, 1042, 1172,1302 ,
1562, 1692, 1952, 2082,2212, 
2342,2472,2602,2732,2862,
2992,3122,3382,3512,3642,
3772]
accu = []
for i in folder:
    test_folder = []
    
    test_folder.append(i)
    train_folder = folder[:]
    train_folder.remove(i)
    label, data= ld.get_data_7(26, train_folder)
    
    label_test, data_test = ld.get_data_7(1, test_folder)
    clf = tr.train(label, data)
    confusion = np.zeros([3,3])
    cnt = 0
    for j in range(0, data_test.shape[0]):
        result = clf.predict(data_test[j,:].reshape(1, -1))
        if label_test[j] == result:
            cnt += 1
        confusion[int(label_test[j]+1), int(result+1)] += 1
   
    accu.append(cnt/data_test.shape[0])
    print("test_folder: ", i, 'train sample: ', label.shape[0], 'test sample: ', label_test.shape[0])
    #print(confusion/np.sum(confusion))     
    print('accuracy: ', accu[-1])
print('all', accu)
print('average: ', np.mean(accu), 'std', np.std(accu))





