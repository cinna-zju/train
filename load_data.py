import sqlite3
import numpy as np
from xml.dom import minidom
import random

folder = [132, 1692, 2082, 2212, 
    2342, 2472, 2602, 2862, 2992, 
    3382, 3512, 3642, 3772] #14*20-3 = 

def get_data(num):
    
    data = np.zeros((1,44))
    label = np.zeros(1)
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
            rate, beg, end = get_time(folder[i]+k)
            for no in range(1,5):
                sql = 'select * from emotions'+str(folder[i]+k)+'_'+str(no) 
                c.execute(sql)

                subdata.append(c.fetchall())
                subdata[no-1] = np.array(subdata[no-1], dtype=np.float32)

                
                # corp video
                mask = subdata[no-1][:,0] > 15
                subdata[no-1] = subdata[no-1][mask, :]
                                
                mask = subdata[no-1][:,0] <  15 + (end - beg)/rate 
                subdata[no-1] = subdata[no-1][mask, :]

                # drop frame when face is lost and calc the percentage
                num_detec[no-1] += sum(subdata[no-1][:,1])
                frame[no-1] += subdata[no-1].shape[0]

                mask = np.array(subdata[no-1][:,1], dtype=np.bool)
                subdata[no-1] = subdata[no-1][mask]

            size = min(subdata[0].shape[0],
                subdata[1].shape[0],
                subdata[2].shape[0],
                subdata[3].shape[0]
            )

            # print(folder[i]+k, 'size: ',size)
            temp = subdata[0][0:size, :]

            for j in range(1, 4):
                temp = np.column_stack((temp, subdata[j][0:size, :]))

            val = temp[:,[9, 20, 31, 42]]
            sublabel = 0.4 * val[:,0] +0.25 * val[:,1] + 0.25 * val[:,2] + 0.1 * val[:,3]
            l = 0

            while l < sublabel.shape[0]:
                if sublabel[l] > 10:
                    sublabel[l] = 1
                else:
                    if sublabel[l] < 0:
                        sublabel[l] = -1
                    else:
                        sublabel[l] = 0
                l += 1 

            label = np.hstack((label, sublabel))
            after = temp
            data = np.row_stack((data, after))

            k += 2

        conn.close()       
        loss = 1 - num_detec/frame
        
    return np.ravel(label[1:]), data[1:,:], loss 
    
def get_time(no):

    beg_smp = []
    end_smp = []
    vidrate = []

    try:
        xmldoc = minidom.parse('./Sessions/'+str(no)+'/session.xml')
        # print('./Sessions/'+str(no)+'/session.xml')
        sess = xmldoc.getElementsByTagName('session')

        for s in sess:
            try:
                beg_smp.append(float(s.attributes['vidBeginSmp'].value))
                end_smp.append(float(s.attributes['vidEndSmp'].value))
                vidrate.append(float(s.attributes['vidRate'].value))
            except KeyError:
                print('invalid xml file')
    except FileNotFoundError:
        print(str(no) + '/session.xml not exist')
            
    return vidrate[0], beg_smp[0], end_smp[0]




