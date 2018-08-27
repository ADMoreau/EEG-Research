# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf 
import os
import csv
import numpy as np
from math import ceil
import time

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
 
def _dtype_feature(ndarray):
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return tf.train.Feature(float_list=tf.train.FloatList(value=ndarray))
        else:  
            raise ValueError("The input should be numpy ndarray. \
                               Instaed got {}".format(ndarray.dtype))

def write_channels(img, label, subject):
    feature = {}
    '''
    for i in range(img.shape[0]):
        s = 'channel ' + str(i)
        i = np.array(img[i], dtype = np.float64)
        feature[s] = _dtype_feature(i)
    '''
    feature['matrix'] = tf.train.Feature(float_list=tf.train.FloatList(value = img.reshape(-1)))
    label = np.array(label, dtype = np.int64)
    feature['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value = [label]))
    feature['subject'] = tf.train.Feature(int64_list=tf.train.Int64List(value = [subject]))
    return feature
            
smalltest = '/media/austin/AustinEXT/tfrecords/1-6test.tfrecords'
smalltrain = '/media/austin/AustinEXT/tfrecords/1-6train.tfrecords'
smallval = '/media/austin/AustinEXT/tfrecords/1-6eval.tfrecords'
bigtest = '/media/austin/AustinEXT/tfrecords/7-9test.tfrecords'
bigtrain = '/media/austin/AustinEXT/tfrecords/7-9train.tfrecords'
bigeval = '/media/austin/AustinEXT/tfrecords/7-9eval.tfrecords'

# Initiating the writer and creating the tfrecords file.

writer1 = tf.python_io.TFRecordWriter(smalltest)
writer2 = tf.python_io.TFRecordWriter(smalltrain)
writer3 = tf.python_io.TFRecordWriter(smallval)
writer4 = tf.python_io.TFRecordWriter(bigtest)
writer5 = tf.python_io.TFRecordWriter(bigtrain)
writer6 = tf.python_io.TFRecordWriter(bigeval)
out = []
a = 0
Action = ''
Subject = ''

for path, dirs, files in os.walk('/media/austin/AustinEXT/CSV'):
    
    for f in files:
        if f.endswith('.csv'):
            temp_array =[]
            
            faa_file_path = os.path.join(path,f)
            print(faa_file_path)
            f = open(faa_file_path, 'r')
            readCSV = csv.reader(f)
            #img = np.genfromtxt(f, delimiter = ',')
            for row in readCSV:
                temp_array.append(row)
            #img = np.array(img.resize((62,250)))
            names = faa_file_path.split("/")
            #print(names)
            Actionold = Action
            Subjectold = Subject
            Subject = int(names[-3])
            Action = str(names[-2])
            if Actionold == Action:
                    a += 1
                    out.append(temp_array)
            elif Actionold != Action and a != 0:
                    #train = ceil(.6 * a)
                    test = int(ceil(.2 * a))
                    eva = 2 * test
                    label = int(Actionold)
                    testout = [out[i][:] for i in range(0,test)]
                    evalout = [out[i][:] for i in range(test,eva)]
                    trainout = [out[i][:] for i in range(eva, a)]
                    out = []
                    time.sleep(2)
                    a = 0
                    if Subjectold == 1 or Subjectold == 2 or Subjectold == 3 or Subjectold == 4 or Subjectold == 5 or Subjectold == 6:

                        for i in testout:
                             feature = write_channels(np.array(i, dtype = np.float64), label, Subjectold)
                             example = tf.train.Example(features=tf.train.Features(feature=feature))
                             writer1.write(example.SerializeToString())
                        testout = []
                        time.sleep(5)
                        print('testout written')
                        for i in trainout:
                             feature = write_channels(np.array(i, dtype = np.float64), label, Subjectold)
                             example = tf.train.Example(features=tf.train.Features(feature=feature))
                             writer2.write(example.SerializeToString())
                        trainout = []
                        time.sleep(5)
                        print('trainout written')
                        for i in evalout:
                             feature = write_channels(np.array(i, dtype = np.float64), label, Subjectold)
                             example = tf.train.Example(features=tf.train.Features(feature=feature))
                             writer3.write(example.SerializeToString())
                        evalout = []
                        time.sleep(5)
                        print('evalout written')
                    elif Subjectold == 7 or Subjectold == 8 or Subjectold == 9:
                        
                        for i in testout:
                             feature = write_channels(np.array(i, dtype = np.float64), label, Subjectold)
                             example = tf.train.Example(features=tf.train.Features(feature=feature))
                             writer4.write(example.SerializeToString())
                        testout = []
                        time.sleep(5)
                        print('testout written')
                        for i in trainout:
                             feature = write_channels(np.array(i, dtype = np.float64), label, Subjectold)
                             example = tf.train.Example(features=tf.train.Features(feature=feature))
                             writer5.write(example.SerializeToString())
                        trainout = []
                        time.sleep(5)
                        print('trainout written')
                        for i in evalout:
                             feature = write_channels(np.array(i, dtype = np.float64), label, Subjectold)
                             example = tf.train.Example(features=tf.train.Features(feature=feature))
                             writer6.write(example.SerializeToString())
                        evalout = []
                        time.sleep(5)
                        print('evalout written')


writer1.close()
writer2.close()
writer3.close()
writer4.close()
writer5.close()
writer6.close()
