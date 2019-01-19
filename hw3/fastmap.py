## Group members :  Myungjin Lee, Krishna Akhil Maddali, Yash Shahapurkar

import numpy as np
import matplotlib.pyplot as plt

fr = open("fastmap-data.txt", "r")
inp = fr.read().splitlines()


def find_dist(obj1,obj2,dist,i,farthest):
    if i == farthest:
        return 0
    for k in range(0,len(inp)):
        if((int(obj1[k]) == i and int(obj2[k]) == farthest) or (int(obj2[k]) == i and int(obj1[k]) == farthest)):
            return dist[k]


obj1=list()
obj2=list()
dist=list()

for i in inp:
    k = i.split("\t")
    obj1.append(int(k[0]))
    obj2.append(int(k[1]))
    dist.append(int(k[2]))

obj1 = np.array(obj1).reshape(len(obj1),1)
obj2 = np.array(obj2).reshape(len(obj2),1)
dist = np.array(dist).reshape(len(dist),1)

def getcoordinate(obj1,obj2,dist):
    
    num_objs = 10
    x=[]
    # get the first pair of farthest objects
    s = np.argmax(dist)
    farthest1 = obj1[s]
    farthest2 = obj2[s]
    dab = float(dist[s])

    # now find distance of all pts from fathest1 and farthest2

    for i in range (1,num_objs+1):
        dai = float(find_dist(obj1,obj2,dist,i,int(farthest1)))
        dbi = float(find_dist(obj1,obj2,dist,i,int(farthest2)))
    
        x.append(((dai*dai) + (dab*dab) - (dbi*dbi))/(2*dab))

    return x


# x is first coordinate
x = getcoordinate(obj1,obj2,dist)
#print("\n The first co-odinates are \n",x)

# update distance metric
newd =[]
for i in range(0,len(inp)):
    #print(i)
    temp1 = (obj1[i][0])
    temp2 = (obj2[i][0])
    tempdist=float(np.sqrt( float((dist[i][0])**2) -(float((x[temp1-1]-x[temp2-1])**2))))
    newd.append( np.array(float(tempdist)))

# y is second coordinate
y = getcoordinate(obj1,obj2,newd)

#print("\n The second co-odinates are \n",y)

final_coord = list(zip(x,y))
print("2D Co-ordinates are",final_coord) 

plt.scatter(x,y)
plt.title('2D embedding')
plt.show()
