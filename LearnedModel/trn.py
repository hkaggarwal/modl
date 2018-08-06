# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:10:24 2018
This is the training code to train the model as described in the following article:
    MoDL: Model-Based Deep Learning Architecture for Inverse Problems
    Link:     https://arxiv.org/abs/1712.02862
@author: haggarwal
"""

# import some libraries
import os,sys,time,importlib
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
from shutil import copyfile
import supportingFunctions as sf
reload(sf) if sys.version_info[0]==2 else importlib.reload(sf)

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True


#--------------------------------------------------------------
#% SET THESE PARAMETERS CAREFULLY

dcGrad='AG' #Either use Automatic gradient (AG) or manual gradient (MG) in data consistency.
nFold=10   #The acceleration factor; e.g. 2-fold acceleration means 50% sampling
nIter=1    #Iterations of alternating strategy as described in Eq. 8   
epochs=100 #number of epoches in tensorflow
saveModelEveryNepoch=50 #save the model after these many epoches

nSlice=200  #Do the training on how many slices. Maximum is 360
batchSize=4 # It is the batch-size. 

sigma=0.01  # the standard deviation of Gaussian noise to be added
cgIter=10   # the number of iterations of conjugate gradient
wtSharing=True  # Whether share the weights across model iterations or not
restoreWeights=False  # Whether restore a previous model for initialization
#if 'restoreWeights' is True then which model to load for initialization.
restoreFromModel='02Aug_0302pm_12C_5L_1R_100E_10F_1V_360S_4B_1W_MG' 
# if you are saving a model multiple time then you can restore a particular checkpoint
# for example, if you saved the model every 50 epoches by using  'SaveModelEveryNepoch'
# then you can use: restoreChkpt='-50'
restoreChkpt='last'

nLayers=5 #number of CNN layers. Each layer is having 64 filters of 3x3 except last.
nCoil=12 # This is fixed number. It is the number of coils used in parallel MRI
#--------------------------------------------------------------------------
#%%Generate a meaningful filename to save the trainined models for testing

start_time=time.time()
saveDir='TFmodels/'
cwd=os.getcwd()
restorePath= cwd+'/TFmodels/'+restoreFromModel
directory=saveDir+datetime.now().strftime("%d%b_%I%M%P_")+str(nCoil)+'C_'+ \
 str(nLayers)+'L_'+str(nIter)+'R_'+str(epochs)+'E_'+str(nFold)+ 'F_'+str(int(sigma*100)) +\
 'V_'+ str(nSlice)+'S_'+str(batchSize)+'B_'+str(int(wtSharing))+'W_'+dcGrad

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName= directory+'/model'

#save all the code in the newly created folder
if dcGrad=='MG':
    import modelMG as mm
    copyfile(cwd+'/modelMG.py',cwd+'/'+directory +'/modelMG.py')    
elif dcGrad=='AG':
    import modelAG as mm
    copyfile(cwd+'/modelAG.py',cwd+'/'+directory +'/modelAG.py')

reload(mm) if sys.version_info[0]==2 else importlib.reload(mm)


copyfile(cwd+'/trn.py',cwd+'/'+directory +'/trn.py')
copyfile(cwd+'/tst.py',cwd+'/'+directory +'/tst.py')
if restoreWeights:
    wts=sf.getWeights(restorePath,restoreChkpt)

#%% save test model
tf.reset_default_graph()

csmT = tf.placeholder(tf.complex64,shape=(None,12,256,232),name='csm')
maskT= tf.placeholder(tf.complex64,shape=(None,256,232),name='mask')
atbT = tf.placeholder(tf.float32,shape=(None,256,232,2),name='atb')

_,out=mm.makeModel(atbT,csmT,maskT,False,nLayers,nIter,nCoil,wtSharing,cgIter)
predTst=out['dc'+str(nIter)]
predTst=tf.identity(predTst,name='predTst')
sessFileNameTst=directory+'/modelTst'

saver=tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    savedFile=saver.save(sess, sessFileNameTst,latest_filename='checkpointTst')
print ('testing model saved:' +savedFile)
#%% read multi-channel dataset

trnOrg,trnAtb,trnCsm,trnMask=sf.getMultiChnlData2('trn',nCoil,nSlice,nFold,sigma)

#%%
tf.reset_default_graph()
csmP = tf.placeholder(tf.complex64,shape=(None,None,None,None),name='csm')
maskP= tf.placeholder(tf.complex64,shape=(None,None,None),name='mask')
atbP = tf.placeholder(tf.float32,shape=(None,None,None,2),name='atb')
orgP = tf.placeholder(tf.float32,shape=(None,None,None,2),name='org')


#%% creating the dataset
nTrn=trnOrg.shape[0]
nBatch= int(np.floor(np.float32(nTrn)/batchSize))
nSteps= nBatch*epochs
nSaveAfter=int(np.floor(np.float32(nTrn*saveModelEveryNepoch)/batchSize))

trnData = tf.data.Dataset.from_tensor_slices((orgP,atbP,csmP,maskP))
trnData = trnData.cache()
trnData=trnData.repeat(count=epochs)
trnData = trnData.shuffle(buffer_size=nSlice)
trnData=trnData.batch(batchSize)
trnData=trnData.prefetch(5)
iterator=trnData.make_initializable_iterator()
orgT,atbT,csmT,maskT = iterator.get_next('getNext')

#%% make training model

_,out=mm.makeModel(atbT,csmT,maskT,True,nLayers,nIter,nCoil,wtSharing,cgIter)
predT=out['dc'+str(nIter)]
predT=tf.identity(predT,name='pred')
loss = tf.reduce_mean(tf.reduce_sum(tf.pow(predT-orgT, 2),axis=0))
tf.summary.scalar('loss', loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#with tf.control_dependencies(update_ops):
optimizer=tf.train.AdamOptimizer().minimize(loss)




#%% training code

print ('*************************************************')
print ('training started at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('parameters are: Epochs:',epochs,' BS:',batchSize,'nSteps:',nSteps,\
'nSave:',nSaveAfter,'nSamples:',nTrn)

saver = tf.train.Saver(max_to_keep=100)
totalLoss,ep=[],0
lossT = tf.placeholder(tf.float32)
lossSumT = tf.summary.scalar("TrnLoss", lossT)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    feedDict={orgP:trnOrg,atbP:trnAtb, maskP:trnMask,csmP:trnCsm}
    sess.run(iterator.initializer,feed_dict=feedDict)

    if restoreWeights:
        sess=sf.assignWts(sess,nLayers,wts)
    savedFile=saver.save(sess, sessFileName)
    print("Model meta graph saved in::%s" % savedFile)
    

    writer = tf.summary.FileWriter(directory, sess.graph)
    for step in tqdm(range(nSteps)):
        try:            
            tmp,_,_=sess.run([loss,update_ops,optimizer])
            totalLoss.append(tmp)
            if np.remainder(step+1,nBatch)==0:
                ep=ep+1
                avgTrnLoss=np.mean(totalLoss)
                lossSum=sess.run(lossSumT,feed_dict={lossT:avgTrnLoss})
                writer.add_summary(lossSum,ep)
                if np.remainder(ep,saveModelEveryNepoch)==0:
                    savedfile=saver.save(sess, sessFileName,global_step=ep,write_meta_graph=True)
                totalLoss=[] #after each epoch empty the list of total loos
        except tf.errors.OutOfRangeError:
            break
    writer.close()

end_time = time.time()
print ('Trianing completed in minutes ', ((end_time - start_time) / 60))
print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('*************************************************')

#%%
