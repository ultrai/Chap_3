import os
os.chdir('/home/mict/2014_Srinivasan')
os.getcwd()
import h5py
import numpy as np

f = h5py.File('/home/mict/2014_Srinivasan/Image.h5', 'r')
f.keys()
I = f['/home/mict/2014_Srinivasan'][...]

f = h5py.File('/home/mict/2014_Srinivasan/grad.h5', 'r')
f.keys()
grad = f['/home/mict/2014_Srinivasan'][...]

f = h5py.File('/home/mict/2014_Srinivasan/result.h5', 'r')
f.keys()
Out = f['/home/mict/2014_Srinivasan'][...]

grad2 = grad.reshape((50,20*5*5))#.swapaxes(2,0).swapaxes(1,0)

grad2_temp=np.absolute(grad2)
grad2_temp = grad2_temp.sum(axis=1)

grad2_up=grad2
grad2_up = grad2_up.sum(axis=1)



I = I[0,:,:]
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
imgplot = plt.imshow(Out[10,:,:]-Out[4,:,:])
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
#for temp in range(Out.shape[0]):
#    Out[temp,:,:] = min_max_scaler.fit_transform(Out[temp,:,:]) 
from matplotlib import image
for temp in range(Out.shape[0]):
    image.imsave(str(temp)+".png", Out[temp,:,:].swapaxes(0,1)) 
n = np.fix(np.sqrt(grad.shape[0]))
for r in range(7):
    for c in range(7):
        if c==0:
            Image = Out[r*c+c,:,:]
        else:
            Image = np.concatenate((Image,Out[r*c+c,:,:]),axis=0)
    if r==0:
       IMAGE = Image
    else:
       IMAGE = np.concatenate((IMAGE,Image),axis=1)   
#IMAGE = -IMAGE
#IMAGE[IMAGE<0]=0
from matplotlib import image
image.imsave("image_example.png", IMAGE.swapaxes(0,1))
image.imsave("image_exampl.png", I[0,:,:].swapaxes(0,1))


import matplotlib.pyplot as plt
imgplot = plt.imshow(Out[10,:,:])
