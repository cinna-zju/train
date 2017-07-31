import numpy as np 
import load_data as ld
from scipy import io
folder = [2, 132,  262, 392, 522, 
652, 782,912, 1042, 1172,1302 ,
1562, 1692, 1952, 2082,2212, 
2342,2472,2602,2732,2862,
2992,3122,3382,3512,3642,
3772]



label, data = ld.get_data_7(27, folder)

io.savemat('./data/feature.mat', {'feature': data, 'label':label})
