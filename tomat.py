import sqlite3
import numpy as np
from scipy import io
folder = [2, 132,  262, 392, 522, 1042,
652, 3122,782, 1302,912, 
1172, 1562, 1692, 1952, 2082,
2212, 2342,2472, 2602, 2732,
2862, 2992, 3382, 3512, 3642,
3772] #
# 540 - 3 -4 -6 =527
data = np.zeros((1,13))
label = []
num_detec = np.zeros(4)
frame = np.zeros(4)
cnt = 0

for i in range(0, 27):
    conn = sqlite3.connect('./data/'+str(folder[i]))
    c = conn.cursor()
    k = 0
    limit = 40
    # if folder[i] == 1952 or folder[i]==1822:
    #     limit = 32
    if folder[i]==262: #or folder[i]==1822:
        limit = 34
    if folder[i]==1952:
        limit = 32
    if folder[i]==1042:
        limit = 28
    while k < limit:
        subdata = []
        size = []
        
        for no in range(1,5):
            
            sql = 'select * from emotions'+str(folder[i]+k)+'_'+str(no) 
            c.execute(sql)

            subdata.append(c.fetchall())
            subdata[no-1] = np.array(subdata[no-1], dtype=np.float32)
        
    
        print('start', i, folder[i])
        io.savemat('./data/mat/'+str(folder[i]+k)+'.mat', {'FR':subdata[0], 'BR':subdata[1], 'BL':subdata[2], 'TO':subdata[3]})
        print('end', i, folder[i])
        k+=2
