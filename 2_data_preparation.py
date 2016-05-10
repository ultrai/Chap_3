from __future__ import division
import os
from glob import glob
import numpy as np
import scipy 
import scipy.misc
from sklearn.cross_validation import StratifiedShuffleSplit
import h5py

print(os.getcwd() + "\n")
os.chdir('/home/mict/2014_Srinivasan')   # Change current working directory
#paths = glob('*/')
paths = ['1/','2/','3/']
Paths2 = ['1/','2/','3/','4/','5/','6/','7/','8/','9/','10/','11/','12/','13/','14/','15/']
n  = 13
paths2 = Paths2[:n]
I = scipy.misc.imread('1/1/1.png')
Images = np.zeros((np.size(I,0),np.size(I,1),1,1), dtype="uint8" )
Images[:,:,0,0] = I
Labels = np.zeros((1), dtype="uint8" )+1
L = np.zeros((1), dtype="uint8" )+1
k=0
for fol in range(len(paths)):
    for sub in range(len(paths2)):
        k=k+1
        files = glob(paths[fol]+paths2[sub]+'*.png')
        I = scipy.misc.imread(files[0])
        images_data = np.zeros((np.size(I,0),np.size(I,1),1,len(files)), dtype="uint8" )
        #Labels = np.concatenate((Labels,np.zeros(len(files))+k),axis=0)
        Labels = np.concatenate((Labels,np.zeros(len(files))+fol+1),axis=0)
        for fil in range(len(files)):
            #images_data[:,:,0,fil] = scipy.misc.imread(files[0])
            images_data[:,:,0,fil] = scipy.misc.imread(paths[fol]+paths2[sub]+str(fil+1)+".png") 
        Images = np.concatenate((Images,images_data),axis=3)    
Images = np.swapaxes(Images,2,1)
Images = np.swapaxes(Images,3,0)

Images = Images.astype(int)
Labels = Labels.astype(int)
X_train = Images[1::,:,:,:]
y_train = Labels[1::]
###############################   test
paths2 = Paths2[n::]
I = scipy.misc.imread('1/1/1.png')
Images = np.zeros((np.size(I,0),np.size(I,1),1,1), dtype="uint8" )
Images[:,:,0,0] = I
Labels = np.zeros((1), dtype="uint8" )+1
L = np.zeros((1), dtype="uint8" )+1
k=0
for fol in range(len(paths)):
    
    for sub in range(len(paths2)):
        k=k+1
        files = glob(paths[fol]+paths2[sub]+'*.png')
        I = scipy.misc.imread(files[0])
        images_data = np.zeros((np.size(I,0),np.size(I,1),1,len(files)), dtype="uint8" )
        #Labels = np.concatenate((Labels,np.zeros(len(files))+k),axis=0)
        Labels = np.concatenate((Labels,np.zeros(len(files))+fol+1),axis=0)
        for fil in range(len(files)):
            images_data[:,:,0,fil] = scipy.misc.imread(files[fil])
        Images = np.concatenate((Images,images_data),axis=3)    
Images = np.swapaxes(Images,2,1)
Images = np.swapaxes(Images,3,0)

Images = Images.astype(int)
Labels = Labels.astype(int)
X_test = Images[1::,:,:,:]
y_test = Labels[1::]
Images = np.concatenate((X_train,X_test),axis=0)
Labels = np.concatenate((y_train,y_test),axis=0)
"""
sss = StratifiedShuffleSplit(Labels, 1, test_size=0.15, random_state=0)   
for train_index, test_index in sss:
    X_train, X_test = Images[train_index], Images[test_index]
    y_train, y_test = Labels[train_index], Labels[test_index]
"""

m = np.mean(X_train,axis=0)
for temp in range(X_train.shape[0]):
    X_train[temp,:,:] = np.subtract(X_train[temp,:,:],m)
s = np.std(X_train,axis=0)
for temp in range(1,X_train.shape[0]):
    X_train[temp,:,:] = np.divide(X_train[temp,:,:],s)

test_2_images = X_test[y_test==2,:,:,:]
test_3_images = X_test[y_test==3,:,:,:]
for temp in range(1,X_test.shape[0]):
    X_test[temp,:,:] = np.subtract(X_test[temp,:,:],m)
for temp in range(1,X_test.shape[0]):
    X_test[temp,:,:] = np.divide(X_test[temp,:,:],s)
test_2 = X_test[y_test==2,:,:,:]
test_3 = X_test[y_test==3,:,:,:]

import random
r =range(X_train.shape[0])
random.shuffle(r)
X_train = X_train[r,:,:,:]
y_train = y_train[r]
"""
if os.path.isfile('Data.hdf5'):         
    os.remove('Data.hdf5')
    os.remove('Data_test.hdf5')
    os.remove('data.h5')
    os.remove('mean.png')
    os.remove('std.png')
"""
with h5py.File('data.h5', 'w') as hf:
    hf.create_dataset('X_train', data=X_train)
    hf.create_dataset('X_test', data=X_test)
    hf.create_dataset('y_train', data=y_train)
    hf.create_dataset('y_test', data=y_test)
    hf.create_dataset('test_2', data=test_2)
    hf.create_dataset('test_3', data=test_3)
    hf.create_dataset('test_2_images', data=test_2_images)
    hf.create_dataset('test_3_images', data=test_3_images)
    hf.create_dataset('m', data=m)                        
    hf.create_dataset('s', data=s)

    
with h5py.File("Data.hdf5", "w") as f:
     dset = f.create_dataset("data", data = X_train, dtype='float32')
     dset = f.create_dataset("label", data = y_train, dtype='float32')
with h5py.File("Data_test.hdf5", "w") as f:
     dset = f.create_dataset("data", data = X_test, dtype='float32')
     dset = f.create_dataset("label", data = y_test, dtype='float32')

from matplotlib import image
image.imsave("imean.png", m[0,:,:])
image.imsave("std.png", s[0,:,:])
"""
from sklearn import preprocessing           
min_max_scaler = preprocessing.MinMaxScaler()
m = min_max_scaler.fit_transform(m[0,:,:])
s = min_max_scaler.fit_transform(s[0,:,:])
import scipy
scipy.misc.imsave('mean.png', m)
scipy.misc.imsave('std.png', s)
"""
################################################  train 
"""
os.system("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64")
os.system("../caffe/build/tools/caffe train -solver solver.prototxt")

"""
"""
import numpy as np
import sys
sys.path.insert(0,  '/home/mict/caffe/python')
import caffe
caffe.set_mode_cpu()#caffe.set_mode_gpu()
#caffe.set_device(1)
solver = caffe.AdaGradSolver('solver.prototxt')
solver.step(500)

################################################  eval
import numpy as np
import sys
sys.path.insert(0,  '/home/mict/caffe/python')
import caffe
net = caffe.Net('train_val_deploy.prototxt', 'train_iter_250.caffemodel', caffe.TEST)
Y_hat = np.zeros((X_test.shape[0],))
for temp in range(X_test.shape[0]):
    in_ = np.array(X_test[temp,:,:,:], dtype=np.float32)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    Y_hat[temp] = net.blobs['prob'].data[0].argmax(axis=0)

"""
