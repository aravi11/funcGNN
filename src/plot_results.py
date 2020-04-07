import matplotlib as mpl
import csv
import sys
mpl.use('TkAgg')
bool = True
args = sys.argv[1]
x = []
y = []

if args == '--train':
    bool = True
elif args =='--test':
    bool = False
else:
    bool = True

if(bool):
    with open('../outputFiles/train_error_graph.txt','r') as error_graph:
        plots = csv.reader(error_graph, delimiter=',')
        for row in plots:
            x.append(int(row[0]))
            y.append(float(row[1]))
        title = 'Training Dataset Size - 40500'
else:
    with open('../outputFiles/test_error_graph.txt','r') as error_graph:
        plots = csv.reader(error_graph, delimiter=',')
        for row in plots:
            x.append(int(row[0]))
            y.append(float(row[1]))
            title = 'Testing Dataset Size - 10125'

import matplotlib.pyplot as plt
plt.plot(x,y, label='Loss function')

plt.ylabel('Error Rate')
plt.xlabel('Epochs')
plt.title(title)
plt.legend()
plt.show()
