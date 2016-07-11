import os
os.chdir('/home/mict/Desktop/retina2')
os.getcwd()
import h5py
import numpy as np
from skimage.transform import rescale
from matplotlib import image
import matplotlib.pyplot as plt
from sklearn import preprocessing
from skimage import io

min_max_scaler = preprocessing.MinMaxScaler()
resp = {}
for temp in range(21):
    f = h5py.File('error/resp_'+str(temp+1)+'.h5', 'r')
    f.keys()
    #resp[temp] = f['/home/raj/Data/'][...]
    res = f['/home/raj/Data/'][...]
    image.imsave(str(temp)+".png", res) 
#    out_0 = np.multiply(-res,-res>0)
#    out_1 = np.multiply(res,res>0)
#    out_0 = (min_max_scaler.fit_transform(out_0)).reshape((out_0.shape[0],out_0.shape[1],1)) 
#    out_1 = (min_max_scaler.fit_transform(out_1)).reshape((out_0.shape[0],out_0.shape[1],1) ) 
#    res =  np.concatenate((out_0,0.5*(out_0+out_1),out_1),2)
#    io.imsave(str(temp)+".png", res/res.max())

temp=17
f = h5py.File('error/resp_'+str(temp+1)+'.h5', 'r')
f.keys()
res = f['/home/raj/Data/'][...]
plt.imshow(res)
plt.colorbar()
