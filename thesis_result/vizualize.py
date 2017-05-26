import os
os.chdir('/home/mict/retina2_desktop/')
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
"""
for Idx in range(1,150):
    res = sio.loadmat('grads_AMD/image_'+str(Idx)+'.mat')['x']
    image.imsave('responses_AMD/'+str(Idx)+"I_1.png", res.swapaxes(0,2).swapaxes(0,1))     
for Idx in range(1,150):
    res = sio.loadmat('grads_DME/image_'+str(Idx)+'.mat')['x']
    image.imsave('responses_DME/'+str(Idx)+"I_1.png", res.swapaxes(0,2).swapaxes(0,1))     
for Idx in range(1,150):
    res = sio.loadmat('grads_Normal/image_'+str(Idx)+'.mat')['x']
    image.imsave('responses_Normal/'+str(Idx)+"I_1.png", res.swapaxes(0,2).swapaxes(0,1))     
"""
# DME image 
Idx = 6
res = sio.loadmat('grads_DME/image_'+str(Idx)+'.mat')['x']
plt.imshow(res.swapaxes(0,2).swapaxes(0,1))
plt.colorbar()

#res = sio.loadmat('grads_DME/resp_'+str(Idx)+'_1.mat')['x']
#plt.imshow(res)
#plt.colorbar()



# Normal
res = sio.loadmat('grads_Normal/image_'+str(1)+'.mat')['x']
plt.imshow(res.swapaxes(0,2).swapaxes(0,1)[48:-48,48:-48,:])
plt.colorbar()

res = sio.loadmat('grads_Normal/Resp_'+str(1)+'_1.mat')['x']
plt.imshow((res[:,24:-24,24:-24]).max(0))
plt.colorbar()

res = sio.loadmat('grads_Normal/resp_'+str(1)+'_1.mat')['x']
plt.imshow(res[24:-24,24:-24])
plt.colorbar()

res = sio.loadmat('grads_Normal/Resp_'+str(1)+'_21.mat')['x']
plt.imshow(res.max(0))
plt.colorbar()

res = sio.loadmat('grads_Normal/resp_'+str(1)+'_21.mat')['x']
plt.imshow(res)
plt.colorbar()



# AMD
res = sio.loadmat('grads_AMD/image_'+str(2)+'.mat')['x']
plt.imshow(res.swapaxes(0,2).swapaxes(0,1)[48:-48,48:-48,:])

res = sio.loadmat('grads_AMD/Resp_'+str(2)+'_1.mat')['x']
plt.imshow((res[:,24:-24,24:-24]).max(0))
plt.colorbar()

res = sio.loadmat('grads_AMD/resp_'+str(2)+'_1.mat')['x']
plt.imshow(res[24:-24,24:-24])
plt.colorbar()

res = sio.loadmat('grads_AMD/Resp_'+str(2)+'_21.mat')['x']
plt.imshow(res.max(0))
plt.colorbar()


res = sio.loadmat('grads_AMD/resp_'+str(2)+'_21.mat')['x']
plt.imshow(res)
plt.colorbar()



# DME
res = sio.loadmat('grads_DME/image_'+str(6)+'.mat')['x']
plt.imshow(res.swapaxes(0,2).swapaxes(0,1)[48:-48,48:-48,:])

res = sio.loadmat('grads_DME/Resp_'+str(6)+'_1.mat')['x']
plt.imshow((res[:,24:-24,24:-24]).max(0))
plt.colorbar()

res = sio.loadmat('grads_DME/resp_'+str(6)+'_1.mat')['x']
plt.imshow(res[24:-24,24:-24])
plt.colorbar()

res = sio.loadmat('grads_DME/Resp_'+str(6)+'_21.mat')['x']
plt.imshow(res.max(0))
plt.colorbar()

res = sio.loadmat('grads_DME/resp_'+str(6)+'_21.mat')['x']
plt.imshow(res)
plt.colorbar()
