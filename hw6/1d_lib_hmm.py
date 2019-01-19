#Group members : Krishna Akhil Maddali, Yash Shahapurkar, Myungjin Lee

import numpy as np
from hidden_markov import hmm

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

def emm(l1,u1):
    temp=list()
    for i in range(0,10):
        temp1=list()
        pr=6*i+1
        l=float(format(i*0.7,"0.1f"))
        u=float(format(i*1.3,"0.1f"))
        for j in range(118):
            if (j/10)>=l and (j/10)<=u:
               temp1.append(1/pr)
            else:
                temp1.append(0)
        temp.append(temp1)
    return temp

def inpro(grid1):
    ip =[]
    s=np.sum(grid1)
    for i in grid1:
        if(i==1):
           ip.append(1/s)
        else:
           ip.append(0)

    return np.array(ip).reshape(1,10)


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
    return np.array(temp)               # Left,Right,Up,Down

#start_probability = inpro(grid)
#transition_probability = trans(grid)                        #each row is one cell



states = ['0','1','2','3','4','5','6','7','8','9'] #('s', 't')
possible_observation = list()
for i in range(0,118):
    possible_observation.append(str(i/10))
#print(possible_observation)
start_probability = np.matrix(inpro(grid)[0])

transition_probability = np.matrix(trans(grid))
emission_probability = np.matrix(emm(l1,u1))

observation=('6.3','5.6','7.6','9.5','6.0','9.3','8.0','6.4','5.0','3.8','3.3')
test = hmm(states,possible_observation,start_probability,transition_probability,emission_probability)
print(test.viterbi(observation))

