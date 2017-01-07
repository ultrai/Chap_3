import os
os.chdir('/home/kliv/phani/retina2_desktop')
os.getcwd()
import h5py
import numpy as np
from skimage.transform import rescale
from matplotlib import image
import matplotlib.pyplot as plt
from sklearn import preprocessing
from skimage import io
import scipy.io as sio
import scipy.misc as im 
min_max_scaler = preprocessing.MinMaxScaler()
# AMD
images = {2,3,10,17,38,54,60,65,131,141,93,1}
# DME
images = {7,11,13,71,73,84,85,94,95,98,105}

for Idx in images:
    res = sio.loadmat('grads/image_'+str(Idx)+'.mat')['x']
    image.imsave('responses/'+str(Idx)+"I.png", res.swapaxes(0,2).swapaxes(0,1))     
for Idx in images:
    res = sio.loadmat('grads/image_'+str(Idx)+'_1.mat')['x']
    image.imsave('responses/'+str(Idx)+"I_1.png", res[0,:,:])     

for temp in range(1,21):
    res = sio.loadmat('grads/resp_'+str(94)+'_'+str(temp+1)+'.mat')['x']
    image.imsave('responses/'+str(94)+'_'+str(temp)+".png", im.imresize(res[:,:],[224,224])) 
for temp in range(2,21):
    res = sio.loadmat('grads/Resp_'+str(98)+'_'+str(temp+1)+'.mat')['x']
    image.imsave('responses/'+str(98)+'_'+str(temp)+".png", im.imresize(res.max(axis=0),[224,224])) 

for Idx in range(1,150):
    res = sio.loadmat('grads/resp_'+str(Idx)+'_1.mat')['x']
    image.imsave('responses/'+str(Idx)+"I_1.png", im.imresize(res[:,:],[224,224]))   

"""
for Idx in range(1,151):
    res = sio.loadmat('grads/image_'+str(Idx)+'.mat')['x']
    image.imsave('responses/'+str(Idx)+"I.png", res.swapaxes(0,2).swapaxes(0,1))    
"""
for Idx in range(1,151):
    res = sio.loadmat('grads/image_'+str(Idx)+'.mat')['x']
    image.imsave('responses/'+str(Idx)+"I.png", res.swapaxes(0,2).swapaxes(0,1))     

for Idx in images:
    temp =2
    res = sio.loadmat('grads/resp_'+str(Idx)+'_'+str(temp+1)+'.mat')['x']
    image.imsave('responses/'+str(Idx)+'_'+str(temp)+".png", im.imresize(res,[224,224])) 
Idx= 93
for temp in range(1,21):
    res = sio.loadmat('grads/resp_'+str(Idx)+'_'+str(temp+1)+'.mat')['x']
    image.imsave('responses/000'+str(Idx)+'_'+str(temp)+".png", im.imresize(res,[224,224])) 
     
for Idx in images:
	for temp in range(1,21):
          res = sio.loadmat('grads/resp_'+str(Idx)+'_'+str(temp+1)+'.mat')['x']
          image.imsave('responses/'+str(Idx)+'_'+str(temp)+".png", im.imresize(res,[128,128])) 
          
#    out_0 = np.multiply(-res,-res>0)
          
#    out_1 = np.multiply(res,res>0)
#    out_0 = (min_max_scaler.fit_transform(out_0)).reshape((out_0.shape[0],out_0.shape[1],1)) 
#    out_1 = (min_max_scaler.fit_transform(out_1)).reshape((out_0.shape[0],out_0.shape[1],1) ) 
#    res =  np.concatenate((out_0,0.5*(out_0+out_1),out_1),2)
#    io.imsave(str(temp)+".png", res/res.max())

temp=17
res = sio.loadmat('grads/resp_'+str(Idx)+'_'+str(temp+1)+'.mat')['x']
plt.imshow(res)
plt.colorbar()
plt.imsave('test.png', data, cmap = plt.cm.gray)