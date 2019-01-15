#!/usr/bin/env python3i


#from sklearn import datasets
#iris = datasets.load_iris()
#digits = datasets.load_digits()

from sklearn import tree
#X = [[0, 0], [1, 1]]
#Y = [0, 1]
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(X, Y)

#clf.predict([[2., 2.]])



from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# 자동으로 데이터셋을 분리해주는 함수
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 데이터 표준화 작업
sc = StandardScaler()
sc.fit(X_train)

# 표준화된 데이터셋
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



iris_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
iris_tree.fit(X_train, y_train)

from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

dot_data = export_graphviz(iris_tree, out_file=None, feature_names=['petal length', 'petal width'],
                          class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())





column=[['High', 'Expensive', 'Loud', 'Talpiot', 'No', 'No'],
 ['High', 'Expensive', 'Loud', 'City-Center', 'Yes', 'No'],
 ['Moderate', 'Normal', 'Quiet', 'City-Center', 'No', 'Yes'],
 ['Moderate', 'Expensive', 'Quiet', 'German-Colony', 'No', 'No'],
 ['Moderate', 'Expensive', 'Quiet', 'German-Colony', 'Yes', 'Yes'],
 ['Moderate', 'Normal', 'Quiet', 'Ein-Karem', 'No', 'No'],
 ['Low', 'Normal', 'Quiet', 'Ein-Karem', 'No', 'No'],
 ['Moderate', 'Cheap', 'Loud', 'Mahane-Yehuda', 'No', 'No'],
 ['High', 'Expensive', 'Loud', 'City-Center', 'Yes', 'Yes'],
 ['Low', 'Cheap', 'Quiet', 'City-Center', 'No', 'No'],
 ['Moderate', 'Cheap', 'Loud', 'Talpiot', 'No', 'Yes'],
 ['Low', 'Cheap', 'Quiet', 'Talpiot', 'Yes', 'Yes'],
 ['Moderate', 'Expensive', 'Quiet', 'Mahane-Yehuda', 'No', 'Yes'],
 ['High', 'Normal', 'Loud', 'Mahane-Yehuda', 'Yes', 'Yes'],
 ['Moderate', 'Normal', 'Loud', 'Ein-Karem', 'No', 'Yes'],
 ['High', 'Normal', 'Quiet', 'German-Colony', 'No', 'No'],
 ['High', 'Cheap', 'Loud', 'City-Center', 'No', 'Yes'],
 ['Low', 'Normal', 'Quiet', 'City-Center', 'No', 'No'],
 ['Low', 'Expensive', 'Loud', 'Mahane-Yehuda', 'No', 'No'],
 ['Moderate', 'Normal', 'Quiet', 'Talpiot', 'No', 'No'],
 ['Low', 'Normal', 'Quiet', 'City-Center', 'No', 'No'],
 ['Low', 'Cheap', 'Loud', 'Ein-Karem', 'Yes', 'Yes']]

pred=[ 'Moderate', 'Cheap', 'Loud', 'City-Center', 'No', 'No']
float_column=numpy.zeros(6*22).reshape(22,6)
prediction=numpy.zeros(6).reshape(1,-1)
vals_pre=list(set(pred))
for i,string in enumerate(pred):
    prediction[i]=int(vals_pre.index(string))
prediction=prediction.tolist()
print(prediction)


for j in range(len(float_column)):
    vals=list(set(column[j]))
    for i, string in enumerate(column[j]):
        float_column[j][i] =int(vals.index(string))
print(float_column)

X=float_column.tolist()
Y=["No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes"]
data_feature_names=["Occupied","Prices","Music","Location","VIP", "Favorite Beer"]

import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import collections


# Training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
predict=clf.predict([[2.0, 0.0, 1.0, 4.0, 3.0, 3.0]])
# moderate, cheap, loud, city center, no, no
print(predict)

# Visualize data
dot_data = tree.export_graphviz(clf,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')







###################################################################
column=[['High', 'Expensive', 'Loud', 'Talpiot', 'No', 'No'],
 ['High', 'Expensive', 'Loud', 'City-Center', 'Yes', 'No'],
 ['Moderate', 'Normal', 'Quiet', 'City-Center', 'No', 'Yes'],
 ['Moderate', 'Expensive', 'Quiet', 'German-Colony', 'No', 'No'],
 ['Moderate', 'Expensive', 'Quiet', 'German-Colony', 'Yes', 'Yes'],
 ['Moderate', 'Normal', 'Quiet', 'Ein-Karem', 'No', 'No'],
 ['Low', 'Normal', 'Quiet', 'Ein-Karem', 'No', 'No'],
 ['Moderate', 'Cheap', 'Loud', 'Mahane-Yehuda', 'No', 'No'],
 ['High', 'Expensive', 'Loud', 'City-Center', 'Yes', 'Yes'],
 ['Low', 'Cheap', 'Quiet', 'City-Center', 'No', 'No'],
 ['Moderate', 'Cheap', 'Loud', 'Talpiot', 'No', 'Yes'],
 ['Low', 'Cheap', 'Quiet', 'Talpiot', 'Yes', 'Yes'],
 ['Moderate', 'Expensive', 'Quiet', 'Mahane-Yehuda', 'No', 'Yes'],
 ['High', 'Normal', 'Loud', 'Mahane-Yehuda', 'Yes', 'Yes'],
 ['Moderate', 'Normal', 'Loud', 'Ein-Karem', 'No', 'Yes'],
 ['High', 'Normal', 'Quiet', 'German-Colony', 'No', 'No'],
 ['High', 'Cheap', 'Loud', 'City-Center', 'No', 'Yes'],
 ['Low', 'Normal', 'Quiet', 'City-Center', 'No', 'No'],
 ['Low', 'Expensive', 'Loud', 'Mahane-Yehuda', 'No', 'No'],
 ['Moderate', 'Normal', 'Quiet', 'Talpiot', 'No', 'No'],
 ['Low', 'Normal', 'Quiet', 'City-Center', 'No', 'No'],
 ['Low', 'Cheap', 'Loud', 'Ein-Karem', 'Yes', 'Yes']]

float_column=numpy.zeros(6*22).reshape(22,6)

for j in range(len(float_column)):
    vals=list(set(column[j]))
    for i, string in enumerate(column[j]):
        float_column[j][i] =int(vals.index(string))
print(float_column)

X=float_column.tolist()
Y=["No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes"]
data_feature_names=["Occupied","Prices","Music","Location","VIP", "Favorite Beer"]
predic=[2.0, 0.0, 1.0, 4.0, 3.0, 3.0]

import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import collections


# Training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
predict=clf.predict([[2.0, 0.0, 1.0, 4.0, 3.0, 3.0]])
                     #[ 'Moderate', 'Cheap', 'Loud', 'City-Center', 'No', 'No']
print(predict)

# Visualize data
dot_data = tree.export_graphviz(clf,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')











## https://pythonprogramminglanguage.com/decision-tree/
