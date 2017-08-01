import sqlite3
import numpy as np 
folder = [
1952,2082,2212,2342,2472,
2602,2732,2862,2992,3122,
3382,3512,3642,3772]#19

for sess in folder:
	k = 0
	limit  = 40

	if sess==262:
		limit = 34
	if  sess==1952:
		limit = 32 #1984 xml damaged
	if sess==1042:
		limit = 28

	while k < limit:
		for no in range(1,5):
			conn = sqlite3.connect('./10s/mp4')
			c = conn.cursor()
			sql = 'select * from emotions'+str(sess+k)+ '_'+str(no)
			c.execute(sql)
			
			conn2 = sqlite3.connect('./data/'+str(sess))
			c2 = conn2.cursor()

			subdata = c.fetchall()
			subdata = np.array(subdata, dtype=np.float32)
            # corp video                             
			mask = subdata[:,0] < 10.5
			subdata = subdata[mask, :]
			li = subdata.tolist()
			for j in li:
				try:
					sql2 = 'insert into emotions'+str(sess+k)+'_'+str(no)+' values('+str(j)[1:-1]+')'
					# if no == 4:
					c2.execute(sql2)
				except:
					print('insert error')
					print(sql2)
					
			print(sess+k, no, len(li))
			conn2.commit()
		k+=2