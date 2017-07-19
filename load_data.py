import sqlite3
import numpy as np
from xml.dom import minidom

folder = [2,132,392,522, 782, 912,
    1172, 1562, 1692, 1952, 2082,
    2212, 2342, 2472, 2602, 2862,
    2992, 3382, 3512, 3642, 3772]


def get_data(num):
    
    data = np.zeros((1,44))
    label = np.zeros((1,1))
    
    for i in range(0, num):
        conn = sqlite3.connect(str(folder[i]))
        c = conn.cursor()
        k = 0
        
        while k < 40:
            subdata = []
            emo, beg, end = get_label(folder[i]+k)
            for no in range(1,5):
                sql = 'select * from emotions'+str(folder[i]+k)+'_'+str(no)+' where time<'+str(15+float(end)-float(beg))+' and time >15' 
                c.execute(sql)
                # print(sql)
                subdata.append(c.fetchall())
                subdata[no-1] = np.array(subdata[no-1])
                # print("subdata:", subdata[no-1].shape)

            size = min(subdata[0].shape[0],
                subdata[1].shape[0],
                subdata[2].shape[0],
                subdata[3].shape[0]
            )

            # print("size", size)
            sublabel = np.ones((size,1)) * new_label(float(emo))
            # print(sublabel.shape)
            # print(label.shape)
            label = np.row_stack((label, sublabel))

            temp = subdata[0][0:size, :]
            # time, num, +9


            for j in range(1, 4):
                temp = np.column_stack((temp, subdata[j][0:size, :]))
                # time, +9
        
            data = np.row_stack((data, temp))
            k += 2
        

    return np.ravel(label[1:]), data[1:,:]
    # delete first row
    # data, mat[n,44]
    


def get_label(no):
    #id = []
    emo_tag = []
    beg_smp = []
    end_smp = []

    try:
        xmldoc = minidom.parse('./Sessions/'+str(no)+'/session.xml')
        sess = xmldoc.getElementsByTagName('session')

        for s in sess:
            try:
                emo_tag.append(s.attributes['feltEmo'].value)
                beg_smp.append(float(s.attributes['vidBeginSmp'].value) / 100)
                end_smp.append(float(s.attributes['vidEndSmp'].value) / 100)
                # id.append(s.attributes['sessionId'].value)
            except KeyError:
                print('invalid xml file')
    except FileNotFoundError:
        print(str(no) + '/session.xml not exist')
            
    return emo_tag[0], beg_smp[0], end_smp[0]


def new_label(emo):
    if emo == 4 or emo == 11:
        return 1 #pleasant
    else:
        if emo == 0:
            return 0 # neutral
        else:
            return -1 # unpleasant