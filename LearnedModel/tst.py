"""
Created on Mon May 7th, 2018 at 1:30 PM

@author: haggarwal
"""
import os,sys,importlib
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import supportingFunctions as sf
reload(sf) if sys.version_info[0]==2 else importlib.reload(sf)

cwd=os.getcwd()
tf.reset_default_graph()
#%% Check each value carefully
directory='02Aug_1158am_12C_5L_5R_100E_10F_1V_360S_1B_1W_MG'
chkPointNum='-50'
nFold=10
sigma=0.01
nCoil=12
nSlice=164

#%% get the dataset

tstOrg,tstAtb,tstCsm,tstMask=sf.getMultiChnlData2('tst',nCoil, nSlice,nFold,sigma)

#numpy.vectorize(complex)(Data[...,0], Data[...,1])
normOrg=sf.normalize01( np.sqrt(tstOrg[...,0]**2+tstOrg[...,1]**2))
normAtb=sf.normalize01(np.sqrt(tstAtb[...,0]**2+tstAtb[...,1]**2))
_,_,psnrAtb=sf.accuray(normOrg,normAtb)

nSlice,nCoil,nRow,nCol=tstCsm.shape

#%%
def getRecon(modelDir,chkPointNum):
    tf.reset_default_graph()
    if chkPointNum=='last':
        loadChkPoint=tf.train.latest_checkpoint(modelDir)
    else:
        loadChkPoint=modelDir+'/model'+chkPointNum
    #print 'Testing with Model:'+ loadChkPoint
    rec=np.empty(tstAtb.shape,dtype=np.complex64)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    #config.gpu_options.per_process_gpu_memory_fraction=.2
    #sess=tf.Session(config=config)
    with tf.Session(config=config) as sess:
        new_saver = tf.train.import_meta_graph(modelDir+'/modelTst.meta')
        new_saver.restore(sess, loadChkPoint)
        graph = tf.get_default_graph()
        predT = graph.get_tensor_by_name('predTst:0')
        maskT = graph.get_tensor_by_name('mask:0')
        atbT  = graph.get_tensor_by_name('atb:0')
        csmT  = graph.get_tensor_by_name('csm:0')
        #lam1=[v for v in tf.trainable_variables() if 'lam1' in v.name ][0]
        #print '\nlambda={0:.5f}'.format(sess.run(lam1)[0])
        #sess.run(lam1.assign( (0.,)))
        wts=sess.run(tf.global_variables())
        for i in tqdm(range(nSlice)):
            dataDict={atbT:tstAtb[[i]],maskT:tstMask[[i]],csmT:tstCsm[[i]] }
            rec[i]=sess.run(predT,feed_dict=dataDict)

    rec=rec.squeeze()
    if nCoil==1:
        rec=np.reshape(rec,(nCoil,nSlice,nRow,nCol,2)) #maintain dimensions

    cpxRec=rec[...,0]+rec[...,1]*1j
    normRec=sf.normalize01(np.abs(cpxRec))
    return normRec,wts
#%% run the model
#directory='01Apr_0213am_12C_5L_10R_500E_10F_1V_100S_1B_1W_'
#chkPointNum='-50'
modelDir= cwd+'/TFmodels/'+directory
print( '*************************************************')
normRec,wts=getRecon(modelDir,chkPointNum)
_,_,psnrRec=sf.accuray(normOrg,normRec)
print ('nfold=',nFold)
print ('  ' + 'Noisy ' + 'Rec')
print ('  {0:.2f} {1:.2f}'.format(psnrAtb.mean(),psnrRec.mean()))
print ('*************************************************')
#%% display

num=90
plt.clf()
plt.subplot(131)
plt.imshow(normOrg[num],cmap=plt.cm.gray, clim=(0.0, .8))
plt.axis('off')
plt.title('Original')
plt.subplot(132)
plt.imshow(normRec[num],cmap=plt.cm.gray,clim=(0.0, .8))
plt.title('Recon '+ str(psnrRec[num].round(2)))
plt.axis('off')
plt.subplot(133)
plt.imshow(normAtb[num],cmap=plt.cm.gray,clim=(0.0, .8))
plt.title('Noisy '+str(psnrAtb[num].round(2)))
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=.01)
plt.show()

