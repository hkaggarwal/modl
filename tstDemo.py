"""
Created on Fri Mar 30 12:39:26 2018
it works on a single image
@author: haggarwal
"""
import os,sys,importlib
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import supportingFunctions as sf
reload(sf) if sys.version_info[0]==2 else importlib.reload(sf)

cwd=os.getcwd()
tf.reset_default_graph()

#%% 'num' represent the slice number on which model is run

num=90 # set this value between 0 to 163. There are total testing 164 slices 

#%%Read the data
tstOrg,tstAtb,tstCsm,tstMask=sf.getData('testing',num)
tstAtb1=sf.c2r(tstAtb)
#%% Load existing model and weights. Then do the reconstruction
print ('Now loading the model ...')
modelDir= cwd+'/'+'LearnedModel'
tf.reset_default_graph()
loadChkPoint=tf.train.latest_checkpoint(modelDir)

rec=np.empty(tstAtb.shape,dtype=np.complex64)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    new_saver = tf.train.import_meta_graph(modelDir+'/modelTst.meta')
    new_saver.restore(sess, loadChkPoint)
    graph = tf.get_default_graph()
    predT   =graph.get_tensor_by_name('predTst:0')
    maskT1   =graph.get_tensor_by_name('mask:0')
    atbT=graph.get_tensor_by_name('atb:0')
    csmT1   =graph.get_tensor_by_name('csm:0')
    wts=sess.run(tf.global_variables())
    dataDict={atbT:tstAtb1,maskT1:tstMask,csmT1:tstCsm }
    rec=sess.run(predT,feed_dict=dataDict)

rec=sf.r2c(rec.squeeze())
print('Reconstruction done')

#%% normalize the data for calculating PSNR

print('Now calculating the PSNR (dB) values')
normOrg=sf.normalize01( np.abs(tstOrg))
normAtb=sf.normalize01( np.abs(tstAtb))
_,_,psnrAtb=sf.accuray(normOrg,normAtb)

normRec=sf.normalize01(np.abs(rec))
_,_,psnrRec=sf.accuray(normOrg,normRec)
print ('*****************')
print ('  ' + 'Noisy ' + 'Rec')
print ('  {0:.2f} {1:.2f}'.format(psnrAtb.mean(),psnrRec.mean()))

print ('*****************')

#%% Display the output images
plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, clim=(0.0, .8))
plt.clf()
plt.subplot(131)
plot(normOrg)
plt.axis('off')
plt.title('Original')
plt.subplot(132)
plot(normAtb)
plt.title('Input, PSNR='+str(psnrAtb.round(2)[0])+' dB' )
plt.axis('off')
plt.subplot(133)
plot(normRec)
plt.title('Output, PSNR='+ str(psnrRec.round(2)[0]) +' dB')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=.01)
plt.show()