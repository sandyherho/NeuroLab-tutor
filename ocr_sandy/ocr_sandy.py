#!/usr/bin/python
import os
os.system('pip install neurolab')

import numpy as np
import neurolab as nl


file_input = "../data/letter.data"

jum_data = 50
label_ori = 'adgimno'
jum_label_ori = len(label_ori)

jum_train = int(.9*jum_data)
jum_test = jum_data - jum_train


awal = 6
akhir = -1

data = [] # fitur
labels = [] # label

with open(file_input, 'r') as f:
    for baris in f.readlines():
        nilai = baris.split('\t')
        if nilai[1] not in label_ori:
            continue
        label = np.zeros((jum_label_ori,1))
        label[label_ori.index(nilai[1])] = 1
        labels.append(label)
        
        karakter_saat_ini = np.array([float(x) for x in nilai[awal:akhir]])
        data.append(karakter_saat_ini)
        
        if len(data) >= jum_data:
            break

#print(type(labels), type(data))
data = np.asfarray(data) # float type array
labels = np.array(labels).reshape(jum_data, jum_label_ori)

jum_dim = len(data[0])

nn = nl.net.newff([[0,1] for _ in range(len(data[0]))], [128,16, jum_label_ori])

error_progress = nn.train(data[:jum_train, :], labels[:jum_train, :], 
                          epochs = 10000, show = 100, goal=.01)

pred = nn.sim(data[jum_train:, :])

for i in range(jum_test):
    print('\n Label asli: ', label_ori[np.argmax(labels[i])])
    print('Prediksi: ', label_ori[np.argmax(pred[i])])