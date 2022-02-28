import numpy as np


imfs = []
res = []
with open('D:/data/2007-12baishuihe/EEMD_txt/ZG91imfs.txt','r') as f:
    a1 = f.readlines()
for b1 in a1:
    c1 = b1.split()
    for d1 in c1:
        e1 = float(d1)
        imfs.append(e1)

with open('D:/data/2007-12baishuihe/EEMD_txt/ZG91res.txt','r') as f:
    a2 = f.readlines()
for b2 in a2:
    c2 = b2.split()
    for d2 in c2:
        e2 = float(d2)
        res.append(e2)

imfs=np.array(imfs)
res=np.array(res)