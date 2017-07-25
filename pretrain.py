import numpy as np
import load_data as ld
import sqlite3
import train

def get_label(emo):
    # joy, sad, disgust, neutral, anger, fear, surprise
    # 0,    1,     2,     3,         4,    5,    6
    tag = np.zeros(7)
    if emo == 0:
        tag[2]=1
    if emo == 1:
        tag[4] = 1
    if emo == 2:
        tag[2] = 1
    if emo == 3:
        tag[5] = 1
    if emo == 4 or emo == 11:
        tag[0] = 1
    if emo == 5:
        tag[1] = 1
    if emo == 6:
        tag[6] = 1
    return tag


folder = [2, 132,  392, 522, 782, 
    912, 1172, 1562, 1692, 1952, 
    2082,2212, 2342, 2472, 2602, 
    2732,2862, 2992, 3382, 3512, 
    3642, 3772] #14*20-3 = 

data = np.zeros((1,28))
label = np.zeros(1, 7)
num_detec = np.zeros(4)
frame = np.zeros(4)
for i in range(0, 22):
    conn = sqlite3.connect('./data/'+str(folder[i]))
    c = conn.cursor()
    k = 0
    limit = 40
    if(folder[i] == 1952):
        limit = 32
    
    while k < limit:
        subdata = []
        emo = ld.get_time(folder[i]+k)
        emotag = get_label(emo)
        beg, end = ld.get_t()
        
        size = 1000
        for no in range(1,5):
            try:
                sql = 'select * from emotions'+str(folder[i]+k)+'_'+str(no) 
                c.execute(sql)
            except sqlite3.OperationalError:
                break
            subdata.append(c.fetchall())
            subdata[no-1] = np.array(subdata[no-1], dtype=np.float32)

            if subdata[no-1].shape[0] == 0:
                break

            # # corp video                               
            mask = subdata[no-1][:,0] < (float(end[str(folder[i]+k)]) - float(beg[str(folder[i]+k)]))/256 
            subdata[no-1] = subdata[no-1][mask, :]

            # drop frame when face is lost and calc the percentage
            num_detec[no-1] += sum(subdata[no-1][:,1])
            frame[no-1] += subdata[no-1].shape[0]
            mask = np.array(subdata[no-1][:,1], dtype=np.bool)
            subdata[no-1] = subdata[no-1][mask]

            if size > subdata[no-1].shape[0]:
                size = subdata[no-1].shape[0]
        # print(folder[i]+k, 'size: ',size)
        if size > 0 and size!=1000:
            temp = subdata[0][0:size, 2:9]
            for j in range(1,4):
                temp = np.hstack((temp, subdata[j][0:size, 2:9]))
            
            # drop neutral
            if val_emo != 0:
                mask = np.ones(size, dtype=np.bool)
                for j in range(0,size):
                    if np.sum(temp[j,:]) == 0:
                        mask[j] = False
                temp = temp[mask]
            data = np.vstack((data, temp))
        
            sublabel = np.ones(temp.shape[0]) * tag
            label = np.hstack((label, sublabel))
        k += 2

    conn.close()       
    data = data[1:,:]
    label = label[1:, :]
print(data.shape)
print(label.shape)

print('positive', label[label>0])
print('neutral', label[label = 0])
print('negative', label[label<0])

train.train(label, data)

