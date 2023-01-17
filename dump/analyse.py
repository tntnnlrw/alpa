import json
import argparse
import pandas as pd
import numpy as np
from statistics import mean

# skews = ['10','100','1000']
gpus = ['4','8']
model = {'4':'1.3B','8':'2.6B'}
bs=[32,64,128,256,512,1024]
# bs=bs.reversed()
# methods=['uni','dagc','acc']
methods=['base','reom']
prefix='test'
autodist=[2.65,3.96,6,9.78,14.91,9.29,17.57,24.84,36.19,76.63,115.04,69.36]

count=0
for gpu in gpus:
    table = {}
    for b in reversed(bs):
        batch_size=int(1024/b)
        table[batch_size] = {}
        filename='dump_'+gpu+'gpu_'+str(b)+'bs.txt'
        numbers=np.loadtxt('data/'+filename)
        table[batch_size]['num_nodes']=numbers[0]
        table[batch_size]['num_edges']=numbers[1]
        table[batch_size]['baseline time/s']=numbers[2]
        filename_r='r_'+filename
        numbers=np.loadtxt('data/'+filename_r)
        table[batch_size]['random time/s']=numbers[2]
        table[batch_size]['mem time/s']='> 600'
        table[batch_size]['autodist time/s']=autodist[count]
        count +=1


        # for method in methods:
        #     x=np.loadtxt('./data/'+prefix+'_'+method+'_'+skew+'_'+str(delta)+'.txt') 
        #     table[delta][method]=mean(x[0][-5:-1])*100     
        pdTable = pd.DataFrame(table).round(2).T
    print(f'model: {model[gpu]}')
    print(pdTable.to_markdown())