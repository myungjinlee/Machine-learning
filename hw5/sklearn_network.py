
from sklearn.neural_network import MLPClassifier
import re
import numpy 

train_txt = 'downgesture_train.list'
test_txt = 'downgesture_test.list'

def read_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    return numpy.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                         count=int(width) * int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))

train_data=[]
train_label=[]
for line in open(train_txt, "r").readlines():
    tmp=line.strip()
    train_data.append(read_pgm(tmp))
    if 'down' in line:
        train_label.append(1)
    else:
        train_label.append(0)
train_data=numpy.array(train_data, dtype=float)
train_label=numpy.array(train_label)
print(train_data.shape)  #(184, 30, 32) 
print(train_label.shape) #(184)


X=train_data
y=train_label
clf = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)
clf.predict([[2., 2.], [-1., -2.]])
clf.score(X,y, sample_weight=None)
test=numpy.zeros((2,3,4))
print(test)

def sigmoid(x):
    return (1.0/(1.0+numpy.exp(-x)))



