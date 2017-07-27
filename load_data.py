import sqlite3
import numpy as np
from xml.dom import minidom
import random
import csv
import pretrain as pt
def get_data_7(num, folder):
    
    data = np.zeros((1,13))
    label = []
    num_detec = np.zeros(4)
    frame = np.zeros(4)
    for i in range(0, num):
        conn = sqlite3.connect('./data/'+str(folder[i]))
        c = conn.cursor()
        k = 0
        limit = 40

        if folder[i]==262:
            limit = 34
        if  folder[i]==1952:
            limit = 32 #1984 xml damaged
        if folder[i]==1042:
            limit = 28
        while k < limit:
            subdata = []
            emo = get_time(folder[i]+k)
            beg, end = get_t()
            size = 1000
            for no in range(1,5):
                
                sql = 'select * from emotions'+str(folder[i]+k)+'_'+str(no) 
                c.execute(sql)

                subdata.append(c.fetchall())
                subdata[no-1] = np.array(subdata[no-1], dtype=np.float32)

                # if subdata[no-1].shape[0] == 0:
                #     print(folder[i]+k, "size = 0")
                    
                # corp video                             
                mask = subdata[no-1][:,0] < (float(end[str(folder[i]+k)]) - float(beg[str(folder[i]+k)]))/256 
                subdata[no-1] = subdata[no-1][mask, :]

                # drop frame when face is lost and calc the percentage
                num_detec[no-1] += sum(subdata[no-1][:,1])
                frame[no-1] += subdata[no-1].shape[0]
                # mask = np.array(subdata[no-1][:,1], dtype=np.bool)
                # subdata[no-1] = subdata[no-1][mask]

                if size > subdata[no-1].shape[0]:
                    size = subdata[no-1].shape[0]

            
            # joy, sad, disgust, neutral, anger, fear, surprise
            temp = np.zeros(1)
            
            
            # get percentage of 9 emotion
            e = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float64)

            # emo_7 = subdata[ii][:, 2:9]
            emo_7 = 1.0* subdata[0][0:size,2:9] # + 0.5 * subdata[1][0:size,2:9] + 0.5 * subdata[2][0:size,2:9] + 1 * subdata[3][0:size,2:9]
            # emo_7 = subdata[0][0:size, 2:9]
            
            for j in emo_7:
                t = j[0:7]
                emo_max = np.argmax(t)
                if emo_max != 3:
                    if j[emo_max] != 0:
                        e[np.argmax(j)] += 1#j[emo_max]
                    else:
                        e[3] += 1

            e /= np.sum(e)

            # get val of a session
            if size > 0 and size != 1000:
                val = 1 * 1 * subdata[0][0:size,9] # + 0.1* subdata[1][0:size,9] + 0.1*subdata[2][0:size,9] + 0.4*subdata[3][0:size,9]
                val_arr = np.array([val[val==0].shape[0], val[val>0].shape[0], val[val<0].shape[0]])
                val_arr = val_arr/size

                ega = 1  * subdata[0][0:size,10] # + 0.1* subdata[1][0:size,10] + 0.1*subdata[2][0:size,10] + 0.4*subdata[3][0:size,10]
                ega_arr = np.array([ega[ega > 66].shape[0], 
                    size - ega[ega > 66].shape[0] - ega[ega<33].shape[0],
                    ega[ega<33].shape[0]])
                ega_arr = ega_arr/size
                temp = np.hstack((temp, e))
                temp = np.hstack((temp, val_arr))
                temp = np.hstack((temp, ega_arr))
                temp = temp[1:]
                data = np.vstack((data, temp))
                # mode
                label.append(get_val(emo))
                
            k += 2
        conn.close()       
        loss = 1 - num_detec/frame
        
    return np.array(label), data[1:,:]
    
def get_time(no):

    feltEmo = []
    try:
        xmldoc = minidom.parse('./Sessions/'+str(no)+'/session.xml')
        # print('./Sessions/'+str(no)+'/session.xml')
        sess = xmldoc.getElementsByTagName('session')

        for s in sess:
            try:
                feltEmo.append(float(s.attributes['feltEmo'].value))

            except KeyError:
                print('invalid xml file', no)
    except FileNotFoundError:
        print(str(no) + '/session.xml not exist')
            
    return feltEmo[0]   #, vidrate[0],beg_smp[0], end_smp[0]


def get_val(emo):
    if emo == 4 or emo == 11:
        return 1
    else:
        if emo == 0 or emo ==6:
            return 0
        else:
            return -1

def get_aro(emo):
    if emo == 5 or emo == 2 or emo == 0:
        return 0
    else:
        if emo == 4 or emo == 11:
            return 1
        else:
            return 2

def get_t():
    begin = {}
    end = {}
    with open('begin_end.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            begin[row[0]] = row[2]
            end[row[0]] = row[3]
    
    return begin, end



def get_data_28(num, folder):
    data = np.zeros((1,32))
    label = []
    num_detec = np.zeros(4)
    frame = np.zeros(4)
    for i in range(0, num):
        conn = sqlite3.connect('./data/'+str(folder[i]))
        c = conn.cursor()
        k = 0
        limit = 40

        if(folder[i] == 1952):
            limit = 32
        if(folder[i] == 262):
            limit = 32
        
        while k < limit:
            subdata = []
            result1 = np.zeros(7)
            emo = get_time(folder[i]+k)
            beg, end = get_t()
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

                # corp video                             
                mask = subdata[no-1][:,0] < (float(end[str(folder[i]+k)]) - float(beg[str(folder[i]+k)]))/256 
                subdata[no-1] = subdata[no-1][mask, :]

                # drop frame when face is lost and calc the percentage
                num_detec[no-1] += sum(subdata[no-1][:,1])
                frame[no-1] += subdata[no-1].shape[0]
                mask = np.array(subdata[no-1][:,1], dtype=np.bool)
                subdata[no-1] = subdata[no-1][mask]

                if size > subdata[no-1].shape[0]:
                    size = subdata[no-1].shape[0]

            if size > 0 and size!=1000:
                temp = subdata[0][0:size, 2:9]
                for j in range(1,4):
                    temp = np.hstack((temp, subdata[j][0:size, 2:9]))

                
                e = np.zeros(28)

                for iii in range(0,4):
                    emo_7 = temp[:, iii*7:7+iii*7] 
                
                    for j in emo_7:
                        t = j[0:7]
                        emo_max = np.argmax(t)
                        if emo_max != 3:
                            if j[emo_max] != 0:
                                e[np.argmax(j)] += 1#j[emo_max]
                            else:
                                e[3] += 1

                    e /= np.sum(e)
                a = np.zeros(4)
                for iii in range(0,4):
                    v = subdata[iii][:,7]
                    if v[v>0].shape[0] > v[v<0].shape[0] and v[v>0].shape[0]> 5:
                        a[iii] = 1
                    if v[v>0].shape[0] < v[v<0].shape[0] and v[v<0].shape[0] >5:
                        a[iii] = -1
                e = np.hstack((e, a))
                data = np.vstack((data, e))
                label.append(get_val(emo))


            k += 2
    return np.array(label), data[1:, :]