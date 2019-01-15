# Group members :  Myungjin Lee, Krishna Akhi Maddalil, Yash Shahapurkar

import math
import copy

class Table:
     def __init__(self,inp,depth,label,parent,split,att):
         self.label=label
         self.tabl=inp
         self.d=depth
         self.parent = parent
         self.split = split
         self.att = att
     
     
def DT(table):
    te={}
    et=entropytable(table)
    if(et==0):
        t=[]
        i =table.tabl        
        t.append(table.tabl[0][len(table.tabl[0])-1])      
        return t
    values=[None,-9999]
    le=len(table.label)-1
    for j in range(0,le):               #labels
        i=table.label[j]
        x=entropyattr(table,i)
        if(values[0]==None):
            values[0]=i
            values[1]=et-x
        else:
            if((et-x)>values[1]):
                values[0]=i
                values[1]=et-x
    table.split = values[0]
    sp=values[0]
    if(values[0]==None):
        te=['No']
        return te
        
    t1={} 
    for k in vnam[values[0]]:
        ntable=modifytable(table,k,values[0])
        if(ntable.tabl==[]):
            continue  
        t1[k]=DT(ntable)
    te[values[0]]=t1
    return te    
        
def printout(table,ar):
     array=ar
     if(table.split==None):                            #leaf
          array.append(table.tabl[0][len(table.tabl[0])-1])
          array.append(table.att)
          printout(table.parent,array)
     elif(table.parent==None):                         #root
          array.append(table.split)
          for i in reversed(array):
               print(i)
          return
     else:
          array.append(table.split)
          array.append(table.att)
          printout(table.parent,array)
     return     
          
     
     
     

def modifytable(table,k,labelvalue):
    newtable=[]
    newdepth=table.d+1
    newlabel=copy.deepcopy(table.label)
    indexatt=table.label.index(labelvalue)
    newlabel=newlabel[0:indexatt]+newlabel[indexatt+1:]
    for i in table.tabl:
        if(i[indexatt]==k):
            newtable.append(i[0:indexatt]+i[indexatt+1:])
    temp= Table(newtable,newdepth,newlabel,table,None,k)
    return temp
            
    
    
        
def entropytable(table):
    yes=0
    no=0
    l=len(table.tabl[0])-1
    for i in table.tabl:
        if(i[l]=='Yes'):
            yes+=1
        elif(i[l]=="No"):
            no+=1
    yes=float(yes)
    no=float(no)
    total=yes+no
    if(yes==0):
            entr=-((no/total)*(math.log(no/total,10)))
    elif(no==0):
             entr=-((yes/total)*(math.log(yes/total,10)))
    else:     
            entr=-((yes/total)*(math.log(yes/total,10)))-((no/total)*(math.log(no/total,10)))
    return entr
    
def entropyattr(table,att):
    global vnam
    ent=0
    ind=table.label.index(att)
    l=len(table.tabl[0])-1
    conv=vnam[att]
    Total=len(table.tabl)
    Total=float(Total)
    for j in conv:                  # all possible values for labels
        yes=0
        no=0
        for i in table.tabl:        #i is rows
            if(i[ind]==j):          #index of attribute
                if(i[l]=='Yes'):
                    yes+=1
                elif(i[l]=='No'):
                    no+=1    
                
        yes=float(yes)
        no=float(no)
        total=yes+no
        if(yes==0 and no==0):
            entr=0
        elif(yes==0):
            entr=-((no/total)*(math.log(no/total,10)))
        elif(no==0):
             entr=-((yes/total)*(math.log(yes/total,10)))     
        else:     
            entr=-((yes/total)*(math.log(yes/total,10)))-((no/total)*(math.log(no/total,10)))
        entr=(total/Total)*entr
        ent=ent+entr
    return ent
        
    


'''-------------------MAIN------------------------'''
fr=open("dt-data.txt","r")
inp=fr.read().splitlines()
lbl=inp[0][1:(len(inp[0])-1)]
lbl=lbl.split(", ")
inp=inp[2:]
val=[]
for i in inp:
    val.append(i[4:(len(i)-1)].split(", "))

occupied=['High','Moderate','Low']
price=['Expensive','Normal','Cheap']
music=['Loud','Quiet']
location=['Talpiot', 'City-Center', 'Mahane-Yehuda', 'Ein-Karem', 'German-Colony']
vip=['Yes','No']
favorite_beer=['Yes','No']
enjoy=['Yes','No']
vnam={'Occupied':occupied, 'Price':price, 'Music':music, 'Location':location,'VIP':vip,'Favorite Beer':favorite_beer,'Enjoy':enjoy} 
root=Table(val,1,lbl,None,None,None)
a={}

a['']=(DT(root))
print a['']


