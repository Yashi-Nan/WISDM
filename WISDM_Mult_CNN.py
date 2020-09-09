# multichannel cnn lstm model
import numpy as np
# set the seeds for consistency
seed = 1
np.random.seed(seed)# set the numpy seed before importing keras
import random
random.seed(seed) #set the build-in seed
import tensorflow as tf
tf.random.set_seed(seed) # set the seed for tf
import pandas as pd
from keras.utils.vis_utils import plot_model
from timeit import default_timer as timer
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from numpy import dstack
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers import concatenate
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from plot_confusion_matrix import cm_analysis

# load the labels file as a numpy array
def load_file(filepath):
    labels = np.loadtxt(filepath, str)
    return labels

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = np.loadtxt(prefix + name, ndmin = 2)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded

#load a dataset group, such as train or test
def load_dataset_group(group, prefix = ''):
    #load all 3 files as a single array
    filenames = list()
    # acceleration
    filenames += [group+'_acc_x.txt', group+'_acc_y.txt', group+'_acc_z.txt']
    # load input data
    X = load_group(filenames, prefix)
    # load class output
    y = load_file(prefix + group + 'y.txt') #testy/ trainy
    return X, y

#load the dataset, returns train and test X and y elements
def load_dataset():
    # load all train
    trainX, trainy = load_dataset_group('train', 'WISDM/')
    print('trainX shape: ', trainX.shape, 'trainy shape :', trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', 'WISDM/')
    print('testX shape: ', testX.shape, 'testy shape', testy.shape)
    print('train class',pd.DataFrame(trainy).groupby(0).size())
    return trainX, trainy, testX, testy

trainX, trainy, testX, testy = load_dataset()

# encode class as integers
le = LabelEncoder()
le.fit(trainy)
le.fit(testy)
int_trainy = le.transform(trainy)
int_testy = le.transform(testy)
# convert integers to one hot encode
dum_trainy = np_utils.to_categorical(int_trainy)
dum_testy = np_utils.to_categorical(int_testy)
print('trainX shape: ', trainX.shape, 'trainy one hot shape :', dum_trainy.shape,
      '\ntestX shape: ', testX.shape, 'testy one hot shape: ', dum_testy.shape)

# build a model
def build_model(X, y):

    # channel 1
    inputs1 = Input(shape=(X.shape[1], 1))
    conv1_1 = Conv1D(filters = 32, kernel_size = 3, activation = 'relu')(inputs1)
    conv1_2 = Conv1D(filters = 32, kernel_size = 3, activation = 'relu')(conv1_1)
    flat1 = Flatten()(conv1_2)
    # channel 2
    inputs2 = Input(shape=(X.shape[1], 1))
    conv2_1 = Conv1D(filters = 20, kernel_size = 5, activation = 'relu')(inputs2)
    conv2_2 = Conv1D(filters = 20, kernel_size = 5, activation = 'relu')(conv2_1)
    flat2 = Flatten()(conv2_2)
    # channel 3
    inputs3 = Input(shape=(X.shape[1], 1))
    conv3_1 = Conv1D(filters = 20, kernel_size = 7, activation = 'relu')(inputs3)
    conv3_2 = Conv1D(filters = 20, kernel_size = 7, activation = 'relu')(conv3_1)
    flat3 = Flatten()(conv3_2)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # fully-connected
    dense1 = Dense(40, activation='relu')(merged)
    output = Dense(y.shape[1], activation='softmax')(dense1)
    model = Model(inputs = [inputs1, inputs2, inputs3], outputs = output)

    # save a plot of the model
    # plot_model(model, show_shapes = True, to_file='WISDM_multi-CNN.png')
    model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
    return model

# run an experiment


# fit the model
verbose, epochs, batch_size = 0, 30, 128
trainy_encode = np.argmax(dum_trainy, axis =1) # the column index of max value for each line
weights = class_weight.compute_class_weight('balanced',
                                            np.unique(trainy_encode),trainy_encode)
dict_weights = dict(enumerate(weights))
print('dict_weights:\n',dict_weights)


#start = timer()
#loss, accuracy = list(), list()
# fit the model
model = build_model(trainX, dum_trainy)
start = timer()
history = model.fit([trainX[:, :, :1], trainX[:, :, 1:2], trainX[:, :, 2:]], dum_trainy, epochs=epochs,
                        batch_size=batch_size, class_weight=dict_weights,verbose=verbose)
end = timer()
print('> training time:',end-start)
# evaluate the model
test_loss, test_accuracy = model.evaluate([testX[:,:,:1],testX[:,:,1:2],testX[:,:,2:]],
                                              dum_testy, batch_size=batch_size, verbose=0)
test_accuracy = test_accuracy * 100
print(f'>1: loss={test_loss}, accuracy={test_accuracy}')

# predict the test set
start = timer()
pred = model.predict([testX[:,:,:1],testX[:,:,1:2],testX[:,:,2:]])
end = timer()
print(f'> testing time: {end-start}')
# get the column index of max in each row, then transform to the label names
int_predy = np.argmax(pred, axis = 1)
pred_y = le.inverse_transform(int_predy)

# classification report
from sklearn.metrics import classification_report
target_names = ['Downstairs','Jogging','Sitting','Standing','Upstairs','Walking']
print(classification_report(int_testy, int_predy, target_names = target_names,digits=4))

print(model.summary())

# # save confusion matrix
# cm_analysis(testy, pred_y, filename='WISDM_pics/with_weights/new_WISDM_multi_CNN_CM.png',
#             labels=['Downstairs','Jogging','Sitting',
#                     'Standing','Upstairs','Walking'], figsize=(8,8))

# plt.show()
