import os
import sys # standar lib
import numpy as np
import cv2 # opencv 

file_input = "../data/letter.data"


# set parameters for data visualization
faktor_resize_gambar = 12
awal, akhir = 6, -1
panjang, lebar = 16, 8

# baca setiap baris data dan rescale ke 255 (rgb) hingga kita menghentikan loop pakai ctrl +c
# pake list comprehension, untuk memisahkan baris dengan tab
# reshape dari array 1 dim ke gambar berdimensi 2 kemudian mengatur skalanya dengan 
# menggunakan opencv
with open(file_input, 'r') as f:
    for baris in f.readlines():
        data = np.array([255*float(x) for x in baris.split('\t')[awal:akhir]])
        gambar = np.reshape(data, (panjang, lebar))
        gambar_terskala = cv2.resize(gambar, None,
                                     fx=faktor_resize_gambar, 
                                     fy=faktor_resize_gambar)
        print(baris) # display the character
        cv2.imshow('Gambar', gambar_terskala)
        
        c = cv2.waitKey()
        if c == 27:
            break