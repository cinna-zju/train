import sqlite3

#1958, 1972, 1974, 2082, 3126
conn = sqlite3.connect('./data/ttt')
c = conn.cursor()
#282 284 660 664 670 688 1312 1314 1318
sess =  280
conn2 = sqlite3.connect('./data/262')
c2 = conn2.cursor()

for no in range(1,5):
	sql = 'select * from emotions'+str(sess)+ '_'+str(no)
	c.execute(sql)
	try:
		sql4 = 'drop table emotions'+str(sess)+'_'+str(no)
		c2.execute(sql4)
	except sqlite3.OperationalError:
		print("drop error")
	try:
		sql2 = "create table if not exists emotions" + str(sess) + '_' + str(no)  + '(time unique, nums, joy, sadness, disgust, contempt, anger, fear, surprise, valence, engagement)'
		c2.execute(sql2)
	except:
		print('create error')
	
	for j in c.fetchall():
		try:
			
			sql3 = 'insert into emotions'+str(sess)+'_'+str(no)+' values'+str(j)
			# if no == 4:
				#print(sql3)
			c2.execute(sql3)

		except:
			print('insert error')

	conn2.commit()