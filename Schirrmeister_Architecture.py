#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.signal import butter, lfilter
import numpy as np
import os
import argparse
import pandas as pd
import sys
import time
import pickle
import csv
import math
from matplotlib import pyplot as plt
from random import shuffle
#from gooey import Gooey
from keras.utils import to_categorical
from keras import layers, models, optimizers, utils, callbacks
from keras.models import Sequential, Model
from keras.layers import Concatenate, Reshape, Activation, Add, Flatten, Input, Dense, LSTM, Dropout, BatchNormalization, TimeDistributed, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
from keras.layers.merge import concatenate
import tensorflow as tf
#from sklearn.metrics import confusion_matrix
import itertools
#import seaborn as sn
import pandas as pd


K.set_image_data_format('channels_first')

TRAIN_FILE = "CAPS_1-6train.p"
VALID_FILE = "CAPS_1-6eval.p"
TEST_FILE = "CAPS_1-6test.p"

def readfile(file, subject):
    temp_x = []
    temp_y = []
    with open(file, mode='rb') as f:
        x = 0
        while True:
            try:
                temp = list(pickle.load(f))
                m = [*zip(*temp[0])]
                if len(m) > 0:
                    #Only append to input data if the subject is correct and the class is either baseline or planning/ correction
                    if temp[1] == subject: # and (temp[2] == '1' or temp[2] == '2' or temp[2] == '3' or temp[2] == '8'):
                        temp_x.append(m)
                    temp_y.append(str(int(temp[2])-1))
                    x += 1
                else:
                    pickle.load(f)

            except EOFError:
                break
        f.close()
    return temp_x, temp_y

def load_EEG_eval(subject):
    path = os.path.dirname(os.path.abspath(__file__))
    eval = os.path.join(path + '/data/' + VALID_FILE)
    eX, eY = readfile(eval, subject)
    return eX, eY

def load_EEG(subject):
    path = os.path.dirname(os.path.abspath(__file__))
    trainfile = os.path.join(path + '/data/' + TRAIN_FILE)
    testfile = os.path.join(path + '/data/' + TEST_FILE)
    trX, trY = readfile(trainfile, subject)
    teX, teY = readfile(testfile, subject)

    #shuffle the training and validation sets independently
    #shuffling improves training time
    trTemp = list(zip(trX, trY))
    teTemp = list(zip(teX, teY))
    shuffle(trTemp)
    shuffle(teTemp)
    trX, trY = zip(*trTemp)
    teX, teY = zip(*teTemp)

    return (trX, trY), (teX, teY)

class KerasBatchGenerator(object):

    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
    def generate(self):
        while True:
            x = np.array([])
            if self.current_idx == 0:
                start = 0
                end = (self.current_idx + self.batch_size)
            elif self.batch_size + end >= len(self.x):
                # reset the index back to the start of the data set
                #self.current_idx = 1
                start = 0
                end = self.batch_size
            else:
                start = end
                end = self.batch_size + end
            #Initialize x as a matrix containing every row of all the windows of data with 62 columns(channels)
            #Perform frequency analysis to create 5 matrices of similar size appended together
            #Break this matrix back into batch_size many windows of the initial sampling rate
            x_out = self.x[start: end]
            for i in x_out:
                #i.reshape(1, 250, 62)
                #x[i] = freqAnalysis_bandpass(i, None, None, False)
                temp = np.array(i, dtype = 'float32').reshape(250,62)
                temp = freqAnalysis_bandpass(temp, 0, 0, False)
                if x.size != 0: x = np.append(x, temp, axis = 0)
                else: x = temp
            x = x.reshape(self.batch_size, 5, 250, 62)
            temp_y = np.array(self.y[start:end], dtype = "float32").transpose()
            y = to_categorical(temp_y, num_classes=9)
            self.current_idx += 1
            yield ([x], [y])

class TestGenerator(object):

    def __init__(self, x, y, freq, channel, batch_size, perturb, reset):
        self.x = x
        self.y = y
        self.freq = freq
        self.channel = channel
        self.current_idx = 0
        self.batch_size = batch_size
        self.perturb = perturb
        self.reset = reset
    def generate(self):
        while self.current_idx <= self.reset:
            x = np.array([])
            if self.current_idx == 0:
                start = 0
                end = self.batch_size
            else:
                start = end
                end = self.batch_size + end
            try:
                x_out = self.x[start: end]
            except:
                #print(start, end)
                start = end
                end = end + self.batch_size
                x_out = self.x[start:end]
            for i in x_out:
                temp = np.array(i, dtype = 'float32').reshape(250,62)
                temp = freqAnalysis_bandpass(temp, self.freq, self.channel, self.perturb)
                if x.size != 0: x = np.append(x, temp, axis = 0)
                else: x = temp
            #print(x.shape)
            x = x.reshape(self.batch_size, 5, 250, 62)
            temp_y = np.array(self.y[start: end], dtype = "float32").transpose()
            y = to_categorical(temp_y, num_classes=9)
            self.current_idx += 1
            yield ([x], [y])
        return


def freqAnalysis_bandpass(matrix, freq, channel, perturb):
    '''
    function takes a matrix and returns five layers of similar shape where each layer
    corresponds to a certain neuronal activity frequency
    '''
    eeg = matrix
    fs = 250

    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data, axis=0)
        return y

    # filter the data
    cutoffs = [(0.1,4), (4,8), (8,12), (12,30), (30,49)]
    # 0:delta band 1:theta 2:alpha 3:beta 4:low gamma 5:high gamma
    delta = butter_bandpass_filter(eeg, cutoffs[0][0], cutoffs[0][1], fs, order=4)
    #Schirmirrmeister et al. results suggest threshold frequencies of 4 Hz results in greater accuracies
    theta = butter_bandpass_filter(eeg, cutoffs[1][0], cutoffs[1][1], fs, order=4)
    alpha = butter_bandpass_filter(eeg, cutoffs[2][0], cutoffs[2][1], fs, order=4)
    lgamma = butter_bandpass_filter(eeg, cutoffs[3][0], cutoffs[3][1], fs, order=4)
    hgamma = butter_bandpass_filter(eeg, cutoffs[4][0], cutoffs[4][1], fs, order=4)
    out = np.array([delta,theta,alpha,lgamma,hgamma])
    if perturb:
        out[freq, : , channel] = out[freq, : , channel] * 1.1
    return out

def plot_log(filename, show=False): #change show to true if not running on a server
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:,0] += 1

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()

def create_model(args):
    In = Input((5,250,62), batch_shape=(args.batch_size, 5,250,62))

    model = Convolution2D(25, kernel_size = (10, 1), strides = (1,1), padding='valid')(In)
    model = Convolution2D(25, kernel_size = (1, 62), strides = (1,1), padding='valid')(model)
    model = BatchNormalization()(model)
    model = Activation('elu')(model)
    model = MaxPooling2D(pool_size=(3,1), strides=(1,1))(model)
    model = Convolution2D(50, kernel_size = (10, 1), padding='valid')(model)
    model = BatchNormalization()(model)
    model = Activation('elu')(model)
    model = MaxPooling2D(pool_size=(3,1), strides=(1,1))(model)
    model = Convolution2D(100, kernel_size = (10, 1), padding='valid')(model)
    model = BatchNormalization()(model)
    model = Activation('elu')(model)
    model = MaxPooling2D(pool_size=(3,1), strides=(1,1))(model)
    model = Convolution2D(200, kernel_size = (10, 1), padding='valid')(model)
    model = BatchNormalization()(model)
    model = Activation('elu')(model)
    model = MaxPooling2D(pool_size=(3,1), strides=(1,1))(model)
    model = Flatten()(model)
    model = Dense(9)(model)
    output = Activation("softmax")(model)

    model = Model([In], [output])

    return model

def train(model, data, args):

    (x_train, y_train), (x_test, y_test) = data

    log = callbacks.CSVLogger(args.save_dir + '/' + args.subject + '/log.csv')

    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/' + args.subject + '/tensorboard-logs',
                                batch_size=args.batch_size, histogram_freq=int(args.debug), write_graph=True)

    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/' + args.subject + '/weights-{epoch:02d}.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    train_generator = KerasBatchGenerator(x_train, y_train, args.batch_size)
    validation_generator = KerasBatchGenerator(x_test, y_test, args.batch_size)
    validation_steps=int(len(y_test) / args.batch_size)
    steps_per_epoch=int(len(y_train) / args.batch_size)

    #adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
    sgd = optimizers.SGD(lr=0.0001, momentum = .6, decay=1e-8)

    model.compile(optimizer=sgd,loss='mse',metrics=['accuracy'])

    model.fit_generator(generator=train_generator.generate(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=args.epochs,
                        validation_data=validation_generator.generate(),
                        validation_steps=validation_steps,
                        callbacks=[log, tb, checkpoint])

    model.save_weights(args.save_dir + '/' + args.subject + '/trained_model.h5')

    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    plot_log(args.save_dir + '/' + args.subject + '/log.csv', show=True)

    return model

def test(model, args):
    (x_test, y_test) = load_EEG_eval(args.subject)

    print("Running Baseline Test")
    sgd = optimizers.SGD(lr=0.0001, momentum = .6)

    model.compile(optimizer=sgd,loss='mse',metrics=['accuracy'])

    test_steps_per_epoch = np.math.floor(len(y_test)/args.batch_size)

    validation_generator = TestGenerator(x_test, y_test, 0, 0, args.batch_size, False,test_steps_per_epoch)

    predictions = model.predict_generator(validation_generator.generate(), test_steps_per_epoch)

    predictions = np.argmax(predictions, axis=1)

    y_eval = y_test[:predictions.shape[0]]

    y_eval = list(map(int, y_eval))
    cm = confusion_matrix(y_true=y_eval, y_pred=predictions)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, index = [i for i in "1238"],
                  columns = [i for i in "1238"])
    plt.figure(figsize = (10,7))
    x = sn.heatmap(df_cm, annot=True)
    x = x.get_figure()
    x.savefig('Baseline.png')

    validation_generator = TestGenerator(x_test, y_test, 0, 0, args.batch_size, False,test_steps_per_epoch)
    loss, acc = model.evaluate_generator(validation_generator.generate(),  test_steps_per_epoch)

    plt.close('all')
    f.write('0, 0, {}, {}\n'.format(loss, acc))
    print('Testing Accuracy = ' + acc)

def analysis(model, args):

    sgd = optimizers.SGD(lr=0.0001, momentum = .6, clipnorm=1.)

    model.compile(optimizer=sgd,loss='mse',metrics=['accuracy'])

    for frequency in range(5):
        for channel in range(62):
            validation_generator = TestGenerator(x_test, y_test, frequency, channel, args.batch_size, True, test_steps_per_epoch)
            predictions = model.predict_generator(validation_generator.generate(), test_steps_per_epoch)
            print("Plotting")
            predicted_classes = np.argmax(predictions, axis=1)
            cm = confusion_matrix(y_true=y_eval, y_pred=predicted_classes)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            df_cm = pd.DataFrame(cm, index = [i for i in "1238"],
                          columns = [i for i in "1238"])
            plt.figure(figsize = (10,7))
            x = sn.heatmap(df_cm, annot=True)
            x = x.get_figure()
            x.savefig('{}-{}.png'.format(frequency,channel))

            validation_generator = TestGenerator(x_test, y_test, frequency, channel, args.batch_size, True, test_steps_per_epoch)

            loss, acc = model.evaluate_generator(validation_generator.generate(),  test_steps_per_epoch)

            f.write('{}, {}, {}\n'.format(frequency, channel, loss, acc))
            print(frequency,channel,"loss = " + loss,"accuracy = " + acc)
            plt.close('all')

#@Gooey
def main():
    parser = argparse.ArgumentParser(description="JesusNet")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--training', action='store_true',
                        help="Train the model")
    parser.add_argument('--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--analysis', action='store_true',
                        help="Analyze channels for feature extraction")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--subject', type=str, default = 'S1')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = create_model(args)
    model.summary()

    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if args.training:
        (x_train, y_train), (x_test, y_test) = load_EEG(args.subject)
        train(model, data=((x_train, y_train), (x_test, y_test)), args = args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        #manipulate_latent(manipulate_model, (x_test, y_test), args)

        if args.testing:
            test(model, args)

        elif args.analysis:
            analysis(model, args)

        else:
            print("Please choose testing, training or analysis")

main()
