#Group members : Krishna Akhil Maddali , Yash Shahapurkar, Myungjin Lee
import numpy as np
fread = open('hmm-data.txt',"r")
inp = fread.read().splitlines()
grid=list()
for i in range(2,12):
    temp=inp[i].split(" ")
    temp1=list()
    for j in temp:
        temp1.append(float(j))
    grid.append(temp1)

grid = np.array(grid)
noise=list()
for i in range(24,35):
    temp=" ".join(inp[i].split())
    temp =temp.split(" ")
    temp1=list()
    for j in temp:
            temp1.append(float(j))
    
    noise.append(temp1)
noise = np.array(noise)

def dist(i,j,a,b):
    dis = np.sqrt((a-i)**2 + (b-j)**2)
    return dis

lower_ul=[]
lower_ur=[]
lower_ll=[]
lower_lr=[]
upper_ul=[]
upper_ur=[]
upper_ll=[]
upper_lr=[]
for i in range(10):
    for j in range(10):
        lower_ul.append(float(format(0.7*dist(i,j,0,0),'0.1f')))
        lower_ur.append(float(format(0.7*dist(i,j,0,9),'0.1f')))
        lower_ll.append(float(format(0.7*dist(i,j,9,0),'0.1f')))
        lower_lr.append(float(format(0.7*dist(i,j,9,9),'0.1f')))
        upper_ul.append(float(format(1.3*dist(i,j,0,0),'0.1f')))
        upper_ur.append(float(format(1.3*dist(i,j,0,9),'0.1f')))
        upper_ll.append(float(format(1.3*dist(i,j,9,0),'0.1f')))
        upper_lr.append(float(format(1.3*dist(i,j,9,9),'0.1f')))
        
l1=np.array(lower_ul)
l2=np.array(lower_ur)
l3=np.array(lower_ll)
l4=np.array(lower_lr)
u1=np.array(upper_ul)
u2=np.array(upper_ur)
u3=np.array(upper_ll)
u4=np.array(upper_lr)

def emm(l1,l2,l3,l4,u1,u2,u3,u4,obv):
    emp = [None]*100
    for i in range(100):
        
        if (obv[0]<l1[i] or obv[1]<l2[i] or obv[2]<l3[i] or obv[3]<l4[i]):
            emp[i]=0
        elif (obv[0]>u1[i] or obv[1]>u2[i] or obv[2]>u3[i] or obv[3]>u4[i]):
            emp[i]=0
        else:
            temp1 = (u1[i]-l1[i] +0.1)/0.1
            temp2 = (u2[i]-l2[i] +0.1)/0.1
            temp3 = (u3[i]-l3[i] +0.1)/0.1
            temp4 = (u4[i]-l4[i] +0.1)/0.1
            emp[i]=1/(temp1*temp2*temp3*temp4)

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
            
    return np.array(ip).reshape(10,10)
def convert(x,y):
    if(x>0):
        c = 10*(x) + y
    else:
        c=y
    return c

def revconvert(a):
    x = (int)(a/10)
    y = a%10
    return (x,y)

def trans(grid):
    temp = []
    for i in range(10):
        for j in range(10):
            temp1 = [None]*100
            c=0
            if (j-1<0):
                pass
                
            elif(grid[i][j-1]==1):
                c+=1
                v = convert(i,j-1)
                temp1[v]=-1
            if(j+1>9):
                pass
                
            elif(grid[i][j+1]==1):
                c+=1
                v = convert(i,j+1)
                
                temp1[v]=-1
            if (i-1<0):
                pass
                
            elif(grid[i-1][j]==1):
                c+=1
                v = convert(i-1,j)
                temp1[v]=-1
            if(i+1>9):
                pass
                
            elif(grid[i+1][j]==1):
                c+=1
                v = convert(i+1,j)
                temp1[v]=-1
                
            for k in range(0,len(temp1)):
                if (temp1[k]==-1):
                    temp1[k]=1/c
                elif(temp1[k]==None):
                    temp1[k]=0
            temp.append(temp1)
    return np.array(temp)               # Left,Right,Up,Down

inpr = inpro(grid)
tr = trans(grid)                        #each row is one cell

# viterbi

wr = list()


def vitr(wr,inpr,tr):
    leading = np.ones((100,1))
    tp=inpr.reshape(100,1)
    for i in range(11):
        df = emm(l1,l2,l3,l4,u1,u2,u3,u4,noise[i])
        df=df.reshape(100,1)
        if (i==0):
            
            temp1 = np.multiply(leading,tp)
            temp2 = np.multiply(temp1,df)
            wr=temp2
            leading=wr      # dim 100,1
            tp=tr           # dim 100,100
            back=np.array([None]*100).reshape(100,1)
            
            
        else:
            next_leading=np.zeros((100,1))
            next_back=np.array([None]*100).reshape(100,1)

            for j in range(100):
                
                l=leading[j]
                temp1=np.multiply(l,tp[j]).reshape(100,1)
                
                temp2=np.multiply(temp1,df)
                
                for k in range(100):
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
for i in range(10,-1,-1):
    print(revconvert(fpath[i]))
     
