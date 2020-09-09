# dataset and pre-processing
# follow: http://aqibsaeed.github.io/2016-11-04-human-activity-recognition-cnn/
from pandas import to_numeric
from pandas import read_csv
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

plt.style.use('bmh')

def read_data(file_path):
    column_names= ['user-id','activity','timestamp',
                   'x-axis','y-axis','z-axis']
    data = read_csv(file_path,header=None, names=column_names)
    data['z-axis'] = to_numeric(data['z-axis'].str.replace(';', ''))
    print(data.info())
    return data

def feature_standardize(dataset):
    mu = mean(dataset, axis=0) # mean values of every column
    sigma = std(dataset,axis=0)
    norm = (dataset-mu)/sigma
    return norm

def plot_axis(ax, x, y, title):
    ax.plot(x,y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y)-std(y), max(y)+ std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,figsize=(15,10),sharex=True)

    plot_axis(ax0,data['timestamp'],data['x-axis'],'x-axis')
    plot_axis(ax1,data['timestamp'],data['y-axis'],'y-axis')
    plot_axis(ax2,data['timestamp'],data['z-axis'],'z-axis')
    plt.subplots_adjust(hspace=0.2) # the amount of height reserved for space between subplots
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.9) # the top of the subplots of the figure


dataset = read_data('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
# delete rows with NaN
dataset.dropna(axis=0, how='any',inplace=True)

# check the activity of each subject
gp1 = dataset.groupby(['user-id','activity'])
gp2 = dataset.groupby(['user-id'])
pd.set_option('display.max_columns',None, 'display.max_rows',None,'display.width',1000)
# print the number of data for each activity for each subject
print(gp1.size())
# print the number of activity class each subject perform
print(gp2['activity'].nunique())


# standardize axes values
dataset['x-axis'] = feature_standardize(dataset['x-axis'])
dataset['y-axis'] = feature_standardize(dataset['y-axis'])
dataset['z-axis'] = feature_standardize(dataset['z-axis'])



# plot activities for one subjects
plot_id =[5]
plot_sub = dataset[dataset['user-id'].isin(plot_id)]
for activity in np.unique(plot_sub['activity']):
    subset = plot_sub[plot_sub['activity'] == activity][:200]
    plot_activity(activity,subset)
plt.show()

## select data from 2 subject (with ID: 5,12, 36) as testing data
#filter_id = [5,12,36]
#test_sub = dataset[dataset['user-id'].isin(filter_id)]
## subjects for training data (33 subjects in total)
#train_sub = dataset[-dataset['user-id'].isin(filter_id)]
## check the ID of training subjects
#print(pd.unique(train_sub['user-id']))

# sliding window for data segmentation
def windows(data,size):
    start = 0
    while start < data.count():
        yield int(start), int(start+size)
        start += (size)/2

def segment(data, window_size = 80):
    segments = np.empty((0,window_size, 3))
    labels = np.empty((0))
    for (start, end) in windows(data['id'], window_size):
        x = data["x"][start:end]
        y = data['y'][start:end]
        z = data['z'][start:end]
        if(len(dataset['id'][start:end]) == window_size):
            segments = np.vstack([segments,np.dstack([x, y, z])])
            # [0][0] get the label from ModeResult(mode=['label'],count=[])
            labels = np.append(labels, stats.mode(data['activity'][start:end])[0][0])
    return segments, labels

def segment_signal(data, window_size = 80):
    acc_x = np.empty((0,window_size))
    acc_y = np.empty((0, window_size))
    acc_z = np.empty((0, window_size))
    labels = np.empty((0,1))
    for (start, end) in windows(data['timestamp'], window_size):
        x = data["x-axis"][start:end]
        y = data['y-axis'][start:end]
        z = data['z-axis'][start:end]
        if(len(data['timestamp'][start:end]) == window_size):
            acc_x = np.vstack([acc_x,x])
            acc_y = np.vstack([acc_y,y])
            acc_z = np.vstack([acc_z,z])
            # [0][0] get the label from ModeResult(mode=['label'],count=[])
            labels = np.vstack([labels, stats.mode(data['activity'][start:end])[0][0]])
    return acc_x, acc_y, acc_z, labels


## segment the training and testing data
#train_acc_x, train_acc_y, train_acc_z, trainy = segment_signal(train_sub)
#test_acc_x, test_acc_y, test_acc_z, testy = segment_signal(test_sub)

## save to text file
#np.savetxt('WISDM/train_acc_x.txt',train_acc_x,fmt='%f')
#np.savetxt('WISDM/train_acc_y.txt',train_acc_y,fmt='%f')
#np.savetxt('WISDM/train_acc_z.txt',train_acc_z,fmt='%f')
#np.savetxt('WISDM/test_acc_x.txt',test_acc_x,fmt='%f')
#np.savetxt('WISDM/test_acc_y.txt',test_acc_y,fmt='%f')
#np.savetxt('WISDM/test_acc_z.txt',test_acc_z,fmt='%f')
#np.savetxt('WISDM/trainy.txt',trainy, fmt ='%s')
#np.savetxt('WISDM/testy.txt',testy, fmt ='%s')

## one hot encode activity labels and return the encode array
#trainy = np.asarray(pd.get_dummies(trainy))
#testy = np.asarray(pd.get_dummies(testy))





