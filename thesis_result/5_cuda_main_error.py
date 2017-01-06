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
images = {2,3}
for Idx in range(1,360):
    res = sio.loadmat('grads/image_'+str(Idx)+'.mat')['x']
    image.imsave('responses/'+str(Idx)+"I.png", res.swapaxes(0,2).swapaxes(0,1))     

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