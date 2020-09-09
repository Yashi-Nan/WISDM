import numpy as np
from numpy import vstack
import pandas as pd
from pandas import DataFrame

# load a single file as a numpy array
def load_file(filepath):
    data = np.loadtxt(filepath,str,ndmin=2)
    return data

# summarize the balance of classes in an output variable column
def class_breakdown(data):
    # convert the numpy array into a dataframe
    df = DataFrame(data)
    # group data by the class and count the number of each class
    counts = df.groupby(0).size()
    print(counts)
    # retrieve the number of each class
    print('values', counts.values)
    # summarize
    print('length of counts', len(counts))
    for i in range(len(counts)):
        percent = counts[i]/len(df) * 100
        print(f'Class=',counts.index[i], ' total=',counts.values[i], ' percentage=',percent)

# load all train
trainy = load_file('WISDM/trainy.txt')

# summarize class breakdown
print('Train Dataset')
class_breakdown(trainy)

# load all test
testy = load_file('WISDM/testy.txt')

# summarize class breakdown
print('Test Dataset')
class_breakdown(testy)

# summarize combined class breakdown
print('The Whole Set')
combined = vstack((trainy,testy))
class_breakdown(combined)