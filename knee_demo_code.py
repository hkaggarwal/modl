"""
Created on Thu June 27, 2019
This is the testing code that will load a slice of a Knee image and run the MoDL. 
The trained model is loaded from the subdirectory 'knee_trained_MoDL'

@author: haggarwal
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4,suppress=True)
import tensorflow as tf
import h5py as h5
import supportingFunctions as sf

tf.reset_default_graph()


modelDir='knee_trained_MoDL'

loadChkPoint=tf.train.latest_checkpoint(modelDir)
#%% get the dataset
sf.tic()
with h5.File('knee_demo_data.h5','r') as f:
    tstOrg=f['org'][:]
    tstAtb=f['atb'][:]
    tstCsm=f['csm'][:]
    tstMask=f['mask'][:]

sf.toc()

#%%
tstAtb=sf.c2r(tstAtb)
nImg,nCoil,nRow,nCol=tstCsm.shape
tstRec=np.empty(tstAtb.shape,dtype=np.float32)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    new_saver = tf.train.import_meta_graph(modelDir+'/modelTst.meta')
    new_saver.restore(sess, loadChkPoint)
    graph = tf.get_default_graph()
    predT   =graph.get_tensor_by_name('predTst:0')
    maskT   =graph.get_tensor_by_name('mask:0')
    atbT=graph.get_tensor_by_name('atb:0')
    csmT   =graph.get_tensor_by_name('csm:0')
    for i in range(nImg):
        dataDict={atbT:tstAtb[[i]],maskT:tstMask[[i]],csmT:tstCsm[[i]] }
        tstRec[i]=sess.run(predT,feed_dict=dataDict)

tstAtb=sf.r2c(tstAtb)
tstRec=sf.r2c(tstRec)
#%%
fn=lambda x: sf.normalize01(np.abs(x))
normOrg=fn(tstOrg)
normAtb=fn(tstAtb)
normRec=fn(tstRec)
fn= lambda x: np.rot90( sf.np_crop(x,(320,332)), k=-2,axes=(-1,-2))
normOrg=fn(normOrg)
normAtb=fn(normAtb)
normRec=fn(normRec)

psnrAtb=sf.myPSNR(normOrg,normAtb)
psnrRec=sf.myPSNR(normOrg,normRec)

print ('*************************************************')
print ('  ' + 'Noisy ' + 'Rec')
print ('  {0:.2f} {1:.2f}'.format(psnrAtb.mean(),psnrRec.mean()))
print ('*************************************************')

#%% Display the output images
plot= lambda x: plt.imshow(x,cmap=plt.cm.gray)
plt.clf()
plt.subplot(131)
plot(normOrg)
plt.axis('off')
plt.title('Original')
plt.subplot(132)
plot(normAtb)
plt.title('Input, PSNR='+str(psnrAtb.round(2))+' dB' )
plt.axis('off')
plt.subplot(133)
plot(normRec)
plt.title('Output, PSNR='+ str(psnrRec.round(2)) +' dB')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=0)
plt.show()
