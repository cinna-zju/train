import sqlite3
import csv
import numpy as np
import load_data as ld

folder = [2, 132,  262, 392, 522, 652, 3122,
    782, 1302,912, 1172, 1562, 
    1692, 1952, 2082,2212, 2342, 
    2472, 2602, 2732,2862, 2992, 
    3382, 3512, 3642, 3772] #

f = open('valence.csv', 'w')
writer = csv.writer(f)

for i in folder:
    conn = sqlite3.connect('./data/'+str(i))
    c = conn.cursor()
    k = 0
    limit = 40
    if i == 1952:
        limit = 32
    if i == 262:
        limit = 34

    while k < limit:
        size =1000
        subdata = []
        beg, end = ld.get_t()

        for no in range(1,5):
            try:
                sql = 'select * from emotions'+str(i+k)+'_'+str(no)
                c.execute(sql)
            except sqlite3.OperationalError:
                break
            subdata.append(c.fetchall())
            subdata[no-1] = np.array(subdata[no-1], dtype=np.float32)
            if subdata[no-1].shape[0] == 0:
                print(i+k, '0')
                break
            # corp video                             
            mask = subdata[no-1][:,0] < (float(end[str(i+k)]) - float(beg[str(i+k)]))/256 
            subdata[no-1] = subdata[no-1][mask, :]
            
            if size > subdata[no-1].shape[0]:
                size = subdata[no-1].shape[0]
            
        
        try:
            val = 0.4*subdata[0][0:size,9]+0.25*subdata[1][0:size,9]+0.25*subdata[2][0:size,9]+0.1*subdata[3][0:size,9]
            print(i+k, val.shape)
            meta = np.array([i+k, val.shape[0]])
            line = np.hstack((meta, val))
            # print(line.shape)
            writer.writerow(line)

        except IndexError:
            print('error', i, k)
        k+=2


