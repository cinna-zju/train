import sqlite3
import numpy as np
from xml.dom import minidom
import random
import csv

def get_data(num, folder):
    
    data = np.zeros((1,7))
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
        
        while k < limit:
            subdata = []
            emo = get_time(folder[i]+k)
            beg, end = get_t()
            label.append(get_val(emo))
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
                # mask = subdata[no-1][:,0] > 0
                # subdata[no-1] = subdata[no-1][mask, :]                                
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
            
            # joy, sad, disgust, neutral, anger, fear, surprise
            temp = np.zeros(1)
            
            # get percentage of 9 emotion
            e = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
            if size > 0 and size != 1000:
                # emo_7 = subdata[ii][:, 2:9]
                emo_7 = 0.4 * subdata[0][0:size,2:9] + 0.25 * subdata[1][0:size,2:9] + 0.25 * subdata[2][0:size,2:9] + 0.1 * subdata[3][0:size,2:9]
                # emo_7 = subdata[0][0:size, 2:9]
                for j in emo_7:
                    t = j[0:7]
                    emo_max = np.argmax(t)
                    if emo_max != 3:
                        if j[emo_max] != 0:
                            e[np.argmax(j)] += 1
                        else:
                            e[3] += 1

                e /= np.sum(e)

            temp = np.hstack((temp, e))
            temp = temp[1:]
            
            
            data = np.vstack((data, temp))
            k += 2

        conn.close()       
        loss = 1 - num_detec/frame
        
    return np.array(label), data[1:,:], loss 
    
def get_time(no):

    feltEmo = []
    beg_smp = []
    end_smp = []
    vidrate = []

    try:
        xmldoc = minidom.parse('./Sessions/'+str(no)+'/session.xml')
        # print('./Sessions/'+str(no)+'/session.xml')
        sess = xmldoc.getElementsByTagName('session')

        for s in sess:
            try:
                feltEmo.append(float(s.attributes['feltEmo'].value))
                beg_smp.append(float(s.attributes['vidBeginSmp'].value))
                end_smp.append(float(s.attributes['vidEndSmp'].value))
                vidrate.append(float(s.attributes['vidRate'].value))
            except KeyError:
                print('invalid xml file')
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



def get_t():
    begin = {}
    end = {}
    with open('begin_end.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            begin[row[0]] = row[2]
            end[row[0]] = row[3]
    
    return begin, end
        