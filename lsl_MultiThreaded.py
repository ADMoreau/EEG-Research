#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pylsl import StreamInlet, resolve_stream
import numpy as np
from scipy.signal import butter, lfilter
import socket
from datetime import date
import time
from queue import Queue
from threading import Thread
import csv
import pyaudio

class MaxSizeList(object):

    def __init__(self, max_length):
        self.max_length = max_length
        self.ls = []

    def push(self, st):
        if len(self.ls) == self.max_length:
            self.ls.pop(0)
        self.ls.append(st)

    def get_list(self):
        return self.ls[-10000:-1] #was 10000 samples or 20 seconds

# bandpass filter design
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

def waveRatios(mat):

    #sampling rate of the cogniotics cap in Hz
    fs = 500

    cutoffs = [(0.1,4), (4,8), (8,12), (12,30), (30,49)]
    
    # 0:delta band 1:theta 2:alpha 3:beta 4:low gamma -
    delta = butter_bandpass_filter(mat, cutoffs[0][0], cutoffs[0][1], fs, order=4) **2
    theta = butter_bandpass_filter(mat, cutoffs[1][0], cutoffs[1][1], fs, order=4) **2
    alpha = butter_bandpass_filter(mat, cutoffs[2][0], cutoffs[2][1], fs, order=4) **2
    beta = butter_bandpass_filter(mat, cutoffs[3][0], cutoffs[3][1], fs, order=4) **2
    gamma = butter_bandpass_filter(mat, cutoffs[4][0], cutoffs[4][1], fs, order=4) **2

    full = delta + theta + alpha + beta + gamma
    #full = alpha + beta
    
    
    delta = delta/full
    theta = theta/full
    alpha = alpha/full
    beta = beta/full
    gamma = gamma/full
    
    
    return delta, theta, alpha, beta, gamma, full  #outputting only alpa power right now instead of alpha_ratio

def receiveData(q):
    '''
    Create channel to receive EEG data from pylsl
    Save data received to queue
    '''

    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    print("Stream Found")
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    # open file to save data to
    #datetime = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    #outFile = open('Cogniotics_{}.csv'.format(datetime), mode='w')
    #eegWriter = csv.writer(outFile, delimiter = ',')

    while True:
        sample, timestamp = inlet.pull_sample()
        #print(sample)
        eegSample = np.asarray(sample, dtype=float)
        q.push(eegSample)
        #eegWriter.writerow(eegSample)

def sendData(q):
    '''
    Open TCP Socket
    Pull data from the saved list of EEG samples
    Process and send the EEG samples
    '''

    print('Starting up')
    #tcp socket creation
    host = '127.0.0.1'
    port = 7000
    print('Creating Socket')
    #s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s = socket.socket()
    print('Connecting')
    s.connect((host,port))
    print('Connected')
    
    while True:
        eegMatrix = q.get_list()
        delta, theta, alpha, beta, gamma, full = waveRatios(eegMatrix)
        result = alpha[500]*10**5     #remove any excess characters that get left when converted to string/ pulls 500th sample from the middle of the array to cut down on processing issues
        result = str(result.tolist())       #   <-|
        result = result[1:-1]               #  <-|
        s.sendall(result.encode('utf-8'))
        print("sent")
        time.sleep(.05)

    s.close()

def genAudio(q, channel, freq):

    p = pyaudio.PyAudio()

    rate= 44100
    note = freq #Hz
    duration = 1.0

    stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=rate,
                output=True)

    samples = (np.sin(2*np.pi*np.arange(rate*duration)*note/rate)).astype(np.float32)

    while True:
        ls = np.array(q.get_list())
        ch = ls[:, channel]
        delta, theta, alpha, beta, gamma, full = waveRatios(ch)

        volume = alpha[500]*10**4
        print(channel, volume)
        stream.write(volume*samples)

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":

    input_q = MaxSizeList(20000) #Max size used to make sure the list does not get too large for memory

    receiveData_Thread = Thread(target = receiveData, args = (input_q,))
    receiveData_Thread.setDaemon(True)
    receiveData_Thread.start()

    time.sleep(15)
    
    sendData_Thread = Thread(target = sendData, args = (input_q,))
    sendData_Thread.setDaemon(True)
    sendData_Thread.start()

    audioCh_Freq = [(7, 220), (8, 300), (9, 400)]     #Array of channels to use for sound production
    threads = [Thread(target = genAudio, args = (input_q, i[0], i[1])) for i in audioCh_Freq]
    for t in threads:
        t.setDaemon(True)
        t.start()

    receiveData_Thread.join()
    sendData_Thread.join()
    for t in threads:
        t.join()

