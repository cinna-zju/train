import sqlite3
import numpy as np
from xml.dom import minidom

folder = [2,132,392,522, 782, 912,
    1172, 1562, 1692, 1952, 2082,
    2212, 2342, 2472, 2602, 2862,
    2992, 3382, 3512, 3642, 3772]


def get_data(i):
    conn = sqlite3.connect('dataresult')
    c = conn.cursor()
    data = []
    k = 0
    # test now, for use, k < 40 
    while k < 2:
        
        subdata = []
        emo, beg, end = get_label(folder[i]+k)
        
        for no in range(1,5):
            sql = 'select * from emotions'+str(folder[i]+k)+'_'+str(no)+' where time<'+str(end)+' and time>'+str(beg)
            c.execute(sql)
            subdata.append(c.fetchall())
            subdata[no-1] = np.array(subdata[no-1])

        size = min(subdata[0].shape[0],
            subdata[1].shape[0],
            subdata[2].shape[0],
            subdata[3].shape[0]
        )
        for j in range(0,4):
            subdata[j] = subdata[j][0:size-1,:]
    
        data.append(subdata)

        k += 2
    return emo, data




def get_label(no):
    #id = []
    emo_tag = []
    beg_smp = []
    end_smp = []

    try:
        xmldoc = minidom.parse('./session.xml')
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
        print('./new-hci/' + str(no) + '/session.xml not exist')
            
    return emo_tag[0], beg_smp[0], end_smp[0]



emo, data = get_data(0)
print(data[0][0].shape)
                