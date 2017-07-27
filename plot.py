import sqlite3
import numpy as np
from matplotlib import pyplot as plt

conn = sqlite3.connect('./data/'+str(522))
c = conn.cursor()
subdata = []
for no in range(1,5):
    sql = 'select * from emotions526_'+str(no)
    c.execute(sql)

    subdata.append(c.fetchall())
    subdata[no-1] = np.array(subdata[no-1], dtype=np.float32)

for no in range(1,5):
    plt.subplot(4,1,no)
    plt.plot(subdata[no-1][:,0], subdata[no-1][:,2],
        subdata[no-1][:,0], subdata[no-1][:,3],
        subdata[no-1][:,0], subdata[no-1][:,4],
        subdata[no-1][:,0], subdata[no-1][:,5],
        subdata[no-1][:,0], subdata[no-1][:,6],
        subdata[no-1][:,0], subdata[no-1][:,7],
        subdata[no-1][:,0], subdata[no-1][:,8]
    )   
plt.show() 

