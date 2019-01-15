#!/usr/bin/env python3i

import sys
import numpy
import copy

f_obj=open(sys.argv[1],'r')

cate=[]
occup=[]
price=[]
music=[]
locat=[]
vip=[]
favor=[]
enjoy=[]

for line in f_obj:
    tmp=line.split(',')
    if len(tmp)==7:
        if tmp[0].startswith('('):
            cate.append(tmp[0].split('(')[1])
            cate.append(tmp[1].strip())
            cate.append(tmp[2].strip())
            cate.append(tmp[3].strip())
            cate.append(tmp[4].strip())
            cate.append(tmp[5].strip())
            cate.append(tmp[6].split(')')[0].strip())
        else:
            occup.append(str(tmp[0].split()[1]).strip())
            price.append(str(tmp[1]).strip())
            music.append(str(tmp[2]).strip())
            locat.append(str(tmp[3]).strip())
            vip.append(str(tmp[4]).strip())
            favor.append(str(tmp[5]).strip())
            enjoy.append(str(tmp[6].split(';')[0]).strip())

occup=numpy.array(occup)
price=numpy.array(price)
music=numpy.array(music)
locat=numpy.array(locat)
vip=numpy.array(vip)
favor=numpy.array(favor)
enjoy=numpy.array(enjoy)
print('jin')
def get_entropy(e):
    entropy=0
#    ele, count=numpy.unique(i,return_counts=True)
    prob=float(count/len(s))
    for p in prob:
        if p!=0.0:
            entropy=-(p*numpy.log2(p)+(1-p)*numpy.log2(1-p))
    return entropy


mylist=[occup, price, music, locat, vip, favor, enjoy]
for i in mylist:
    print (numpy.unique(i,return_counts=True))
#    bef=get_entropy(i)
    
#    aft=get_entropy(i)
#    if abs(aft-bef)<1e-6:
#        print cate[0]
#    else:

#print(cate[i])
#print numpy.unique(i)[0]+':'+cate[next_root]



f_obj.close()


#str_var=''
#for i in sched:
#    str_var+=str(i)+','
#str_var=str_var[:-1]
#print str(sched[len(sched)-1])+','+str(total)



