# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:29:36 2019

@author: 2008nmj
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
 
filename = 'C:\\Users\\2008nmj\\Downloads\\train-images-idx3-ubyte\\train-images.idx3-ubyte'
binfile = open(filename , 'rb')
buf = binfile.read()
 
index = 0
magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
index += struct.calcsize('>IIII')
 
im = struct.unpack_from('>784B' ,buf, index)
index += struct.calcsize('>784B')
 
im = np.array(im)
im = im.reshape(28,28)
 
fig = plt.figure()
plotwindow = fig.add_subplot(111)
plt.imshow(im , cmap='gray')
plt.show()

im = struct.unpack_from('>784B' ,buf, index)
index += struct.calcsize('>784B')
 
im = np.array(im)
im = im.reshape(28,28)
 
fig = plt.figure()
plotwindow = fig.add_subplot(111)
plt.imshow(im , cmap='gray')
plt.show()