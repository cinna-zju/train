import sqlite3
import numpy as np
from xml.dom import minidom

# folder = [2,132,392,522, 782, 912,
#     1172, 1562, 1692, 1952, 2082,
#     2212, 2342, 2472, 2602, 2862,
#     2992, 3382, 3512, 3642, 3772]

folder = [2, 132, 1692, 1952, 2082, 2212, 
    2342, 2472, 2602, 2862, 2992, 
    3382, 3512, 3642, 3772]

threshold = 30

def get_data(num):
    
    data = np.zeros((1,44))
    label = np.zeros((1,1))
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
            emo, beg, end = get_label(folder[i]+k)
            for no in range(1,5):
                sql = 'select * from emotions'+str(folder[i]+k)+'_'+str(no) 
                c.execute(sql)

                subdata.append(c.fetchall())
                subdata[no-1] = np.array(subdata[no-1], dtype=np.float32)
                # print(sql, subdata[no-1].shape)
                

                if subdata[no-1].shape[0] > 0:
                    # corp video
                    mask = subdata[no-1][:,0] > 15 
                    subdata[no-1] = subdata[no-1][mask, :]
                                    
                    mask = subdata[no-1][:,0] < 15+end-beg 
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

            if emo != 0:
                # EMOMASK = [0,0,1,1,1,1,1,1,1,0,0,
                #     0,0,1,1,1,1,1,1,1,0,0,
                #     0,0,1,1,1,1,1,1,1,0,0,
                #     0,0,1,1,1,1,1,1,1,0,0
                # ]
                # EMOMASK = np.array(EMOMASK, dtype = np.bool)

                # before = temp[:,EMOMASK]
                      
                # mask = np.ones(temp.shape[0], dtype = np.bool)
                
                # # print('before', before.shape)
                # for j in range(0, before.shape[0]):
                #     col = before[j,[]]
                #     # print(col)
                #     for p in col.flat:
                #         if p >= threshold:
                #             break
                #         mask[j] = False

                before = temp[:,[9, 20, 31, 42]]
                mask = np.ones(temp.shape[0], dtype = np.bool)                      
                # print('before', before.shape)
                for j in range(0, before.shape[0]):
                    col = before[j,:]
                    # print(col)
                    for p in col.flat:
                        if p >= threshold or p < 0:
                            break
                        mask[j] = False
                    
                after = temp[mask,:]
                print(folder[i]+k, 'after: ', after.shape)
                sublabel = np.ones((after.shape[0],1)) * new_label(int(emo))
                # print(new_label(int(emo)))
                label = np.row_stack((label, sublabel))
                data = np.row_stack((data, after))

            k += 2
        conn.close()       
        loss = 1 - num_detec/frame
    
    return np.ravel(label[1:]), data[1:,:], loss 
    # delete first row
    # data, mat[n,44]
    
def get_label(no):
    #id = []
    emo_tag = []
    beg_smp = []
    end_smp = []

    try:
        xmldoc = minidom.parse('./Sessions/'+str(no)+'/session.xml')
        # print('./Sessions/'+str(no)+'/session.xml')
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


