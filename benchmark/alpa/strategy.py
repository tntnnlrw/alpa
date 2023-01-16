import os
import time

def write_str(strs):
    with open("test.txt","w") as f:
        f.writelines(strs)
    return

def get_str(lines,indexs):
    for index in indexs:
        assert 'Instruction' in lines[index]
        strs=[]
        strs.append(lines[index])
        i = index+1
        while True:
            if 'Instruction' in lines[i]:
                break
            strs.append(lines[i])
            i += 1
    return strs

def main_():
    f = open('log.txt', 'r')
    lines = f.readlines()
    indexs = []
    for i in range(len(lines)):
        if 'Startegy Map' in lines[i]:
            start_i = i
        if 'Setup device mesh' in lines[i]:
            end_i = i
            break
    for i in range(start_i,end_i):
        if 'dot(' in lines[i]:
            for j in layers:
                if 'layer/'+str(j) in lines[i] and 'remat' not in lines[i]: 
                    indexs.append(i)
                    break
    f.close()
    strs=get_str(lines,indexs)
    write_str(strs)
    return

layers=[0,1]
main_(layers)