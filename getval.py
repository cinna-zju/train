import sqlite3
import csv
import numpy as np
import load_data as ld

folder = [2, 132,  262, 392, 522, 
652, 782,912, 1042, 1172,1302 ,
1562, 1692, 1952, 2082,2212, 
2342,2472,2602,2732,2862,
2992,3122,3382,3512,3642,
3772]

f = open('valence.csv', 'w')
writer = csv.writer(f)

for i in folder:
    conn = sqlite3.connect('./data/'+str(i))
    c = conn.cursor()
    k = 0
    limit = 40
    if i==262:
        limit = 34
    if i==1952:
        limit = 32 #1984 xml damaged
    if i==1042:
        limit = 28

    while k < limit:
        subdata = []

        
        beg, end = ld.get_t()
        size = 1000
        for no in range(1,5):
            sql = 'select * from emotions'+str(i+k)+'_'+str(no) 
            c.execute(sql)
            subdata.append(c.fetchall())
            subdata[no-1] = np.array(subdata[no-1], dtype=np.float32)
            # if subdata[no-1].shape[0] == 0:
            #     print(folder[i]+k, "size = 0")
                
            # corp video                             
            mask = subdata[no-1][:,0] < (float(end[str(i+k)]) - float(beg[str(i+k)]))/256 
            subdata[no-1] = subdata[no-1][mask, :]

        size = 1000
        for no in range(1, 5):
            if subdata[no-1].shape[0] < size:
                size = subdata[no-1].shape[0]
        for no in range(1, 5):
            subdata[no-1] = subdata[no-1][0:size, :]
        
        nums = (subdata[0][:,1]+subdata[1][:,1]+subdata[2][:,1]+subdata[3][:,1]).reshape(-1,1)
        mask = np.array(nums, dtype=np.bool).reshape(-1)
        #print(nums.shape, mask.shape, subdata[0].shape)
        nums = nums[mask,:]
        for no in range(1, 5):
            subdata[no-1] = subdata[no-1][mask, :]
            size = subdata[no-1].shape[0]

        aa = subdata[0][:,:]
        bb = subdata[1][:,:]
        cc = subdata[2][:,:]
        dd = subdata[3][:,:]

        for no in range(1,5):
            # drop frame when face is lost and calc the percentage
            mask = np.array(np.ones(size)-subdata[no-1][:,1], dtype=np.bool)
            subdata[no-1][mask, :] = (aa[mask,:]+bb[mask,:]+cc[mask,:]+dd[mask,:])/nums[mask,:]

        size = subdata[0].shape[0]

        val = 0.4*subdata[0][0:size,9]+0.25*subdata[1][0:size,9]+0.25*subdata[2][0:size,9]+0.1*subdata[3][0:size,9]
        print(i+k, val.shape)
        meta = np.array([i+k, val.shape[0]])
        line = np.hstack((meta, val))
        # print(line.shape)
        writer.writerow(line)

        k+=2


