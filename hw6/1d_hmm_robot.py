#Group members : Krishna Akhil Maddali , Yash Shahapurkar, Myungjin Lee

import numpy as np
fread = open('hmm-data.txt',"r")
inp = fread.read().splitlines()
grid=list()
for i in range(2,12):
    temp=inp[i].split(" ")
    temp1=list()
    for j in range(len(temp)):
        if j==0:
           temp1.append(float(temp[j]))
    grid.append(temp1)

grid = np.array(grid)
noise=list()
for i in range(24,35):
    temp=" ".join(inp[i].split())
    temp =temp.split(" ")
    temp1=list()
    for j in range(len(temp)):
        if j==0:
           temp1.append(float(temp[j]))

    noise.append(temp1)
noise = np.array(noise)


lower_ul=[]
upper_ul=[]
distance=np.array([0,1,2,3,4,5,6,7,8,9],dtype=int)

lower_ul.append(np.multiply(0.7,distance))
upper_ul.append(np.multiply(1.3,distance))
np.around(lower_ul,decimals=1)
np.around(upper_ul,decimals=1)

l1=np.array(lower_ul[0])
u1=np.array(upper_ul[0])

def emm(l1,u1,obv):
    emp = [None]*10
    for i in range(10):
        
        if (obv[0]<l1[i]):
            emp[i]=0
        elif (obv[0]>u1[i]):
            emp[i]=0
        else:
            temp1 = (u1[i]-l1[i] +0.1)/0.1
            emp[i]=1/(temp1)

    return np.array(emp)

def inpro(grid1):
    ip =[]
    s=np.sum(grid1)
    for i in grid1:
        for j in i:
            if(j==1):
                ip.append(1/s)
            else:
                ip.append(0)
            
    return np.array(ip).reshape(10,1)

def trans(grid):
    temp = []
    for i in range(10):
            temp1 = [None]*10
            c=0
            if (i-1<0):
                pass
            elif(grid[i-1]==1):
                c+=1
                temp1[i-1]=-1
            if(i+1>9):
                pass
            elif(grid[i+1]==1):
                c+=1
                temp1[i+1]=-1

            for k in range(0,len(temp1)):
                if (temp1[k]==-1):
                    temp1[k]=1/c
                elif(temp1[k]==None):
                    temp1[k]=0
            temp.append(temp1)
    return np.array(temp)       

inpr = inpro(grid)
tr = trans(grid)                        #each row is one cell

# viterbi

wr = list()


def vitr(wr,inpr,tr):
    leading = np.ones((10,1))
    tp=inpr.reshape(10,1)
    for i in range(11):
        df = emm(l1,u1,noise[i])
        df=df.reshape(10,1)
        #print(df)
        if (i==0):
            
            temp1 = np.multiply(leading,tp)
            temp2 = np.multiply(temp1,df)
            wr=temp2
            leading=wr      # dim 100,1
            tp=tr           # dim 100,100
            back=np.array([None]*10).reshape(10,1)
            
            
        else:
            next_leading=np.zeros((10,1))
            next_back=np.array([None]*10).reshape(10,1)

            for j in range(10):
                
                l=leading[j]
                temp1=np.multiply(l,tp[j]).reshape(10,1)
                
                temp2=np.multiply(temp1,df)
                
                for k in range(10):
                    if(next_leading[k]<temp2[k]):
                        next_leading[k]=temp2[k]
                        next_back[k]=j
                          
            back = np.hstack((back,next_back))
            wr=np.hstack((wr,next_leading))
            leading = next_leading
 
    return wr,back

def path(wr,back):
    path_a=list()
    d = wr[:,10]
    ind = np.argmax(d)
    path_a.append(ind)
    indb=back[ind][10]
    
    for i in range(9,-1,-1):
        ind=indb
        path_a.append(ind)
        
        indb=back[ind][i]

    return path_a
        
        
rtr, back = vitr(wr,inpr,tr)
fpath = path(rtr,back)

print("The path from first position to last position is")
print(fpath[::-1])
     
