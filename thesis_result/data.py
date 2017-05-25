import os
#os.chdir('/home/phani/Data/caffe/examples/retina2')
os.getcwd() 
import glob
import sys

import numpy as np
import time
import scipy.misc as sio
from skimage.transform import resize

for cla in range (1,4):
    for sub in range(1,9):
        dr = ('Digits/Train/'+str(cla)+'/'+str(sub)+'/')
        ls = sorted(glob.glob(dr+'*.png'))
        I_temp = np.zeros((len(ls),3,224,224))
        subject = np.ones(len(ls))*sub
        label = np.zeros(len(ls))+cla
        for x in range(len(ls)):
            I = sio.imread(ls[x])
            I_temp[x,0,:,:] =  resize(I[:,:,0], (224, 224))
            I_temp[x,1,:,:] =  resize(I[:,:,1], (224, 224))
            I_temp[x,2,:,:] =  resize(I[:,:,1], (224, 224))
        if (sub == 1) & (cla==1):
           X_train = I_temp
           Y_train = label
           Subject_train = subject
        else:
           X_train = np.concatenate((X_train,I_temp),axis=0)
           Y_train = np.concatenate((Y_train,label),axis=0)
           Subject_train = np.concatenate((Subject_train,subject),axis=0)
        print(X_train.shape)
for cla in range (1,4):
    for sub in range(9,16):
        dr = ('Digits/Test/'+str(cla)+'/'+str(sub)+'/')
        ls = sorted(glob.glob(dr+'*.png'))
        I_temp = np.zeros((len(ls),3,224,224))
        subject = np.ones(len(ls))*sub
        label = np.zeros(len(ls))+cla
        for x in range(len(ls)):
            I = sio.imread(ls[x])
            I_temp[x,0,:,:] =  resize(I[:,:,0], (224, 224))
            I_temp[x,1,:,:] =  resize(I[:,:,1], (224, 224))
            I_temp[x,2,:,:] =  resize(I[:,:,1], (224, 224))
        if (sub == 9) & (cla==1):
           X_test = I_temp
           Y_test = label
           Subject_test = subject
        else:
           X_test = np.concatenate((X_test,I_temp),axis=0)
           Y_test = np.concatenate((Y_test,label),axis=0)
           Subject_test = np.concatenate((Subject_test,subject),axis=0)
m = np.mean(X_train,axis=0)
for temp in range(X_train.shape[0]):
    X_train[temp,0,:,:] = np.subtract(X_train[temp,0,:,:],m[0,:,:])
    X_train[temp,1,:,:] = np.subtract(X_train[temp,1,:,:],m[1,:,:])
    X_train[temp,2,:,:] = np.subtract(X_train[temp,1,:,:],m[2,:,:])
test_1_images = X_test[Y_test==1]
test_2_images = X_test[Y_test==2]
test_3_images = X_test[Y_test==3]
for temp in range(X_test.shape[0]):
    X_test[temp,0,:,:] = np.subtract(X_test[temp,0,:,:],m[0,:,:])
    X_test[temp,1,:,:] = np.subtract(X_test[temp,1,:,:],m[1,:,:])
    X_test[temp,2,:,:] = np.subtract(X_test[temp,1,:,:],m[2,:,:])
test_1 = X_test[Y_test==1]
test_2 = X_test[Y_test==2]
test_3 = X_test[Y_test==3]
sub_1 = Subject_test[Y_test==1]
sub_2 = Subject_test[Y_test==2]
sub_3 = Subject_test[Y_test==3]

import random
r =range(X_train.shape[0])
random.shuffle(r)
X_train = X_train[r,:,:,:]
Y_train = Y_train[r]

ind = np.ones((1))+25
import h5py
with h5py.File('data.h5', 'w') as hf:
    hf.create_dataset('X_train', data=X_train)
    hf.create_dataset('X_test', data=X_test)
    hf.create_dataset('Y_train', data=Y_train)
    hf.create_dataset('Y_test', data=Y_test)
    hf.create_dataset('m', data=m*255)                        

with h5py.File('data_error.h5', 'w') as hf:
    hf.create_dataset('test_1', data=test_1)
    hf.create_dataset('test_2', data=test_2)
    hf.create_dataset('test_3', data=test_3)
    hf.create_dataset('sub_1', data=sub_1)
    hf.create_dataset('sub_2', data=sub_2)
    hf.create_dataset('sub_3', data=sub_3)
    hf.create_dataset('ind', data=ind)
    hf.create_dataset('test_1_images', data=test_1_images)
    hf.create_dataset('test_2_images', data=test_2_images)
    hf.create_dataset('test_3_images', data=test_3_images)


