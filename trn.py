# -*- coding: utf-8 -*-
"""
This is the training code to train the model as described in the following article:

MoDL: Model-Based Deep Learning Architecture for Inverse Problems
by H.K. Aggarwal, M.P. Mani, M. Jacob from University of Iowa.

Paper dwonload  Link:     https://arxiv.org/abs/1712.02862

This code solves the following optimization problem:

    argmin_x ||Ax-b||_2^2 + ||x-Dw(x)||^2_2

 'A' can be any measurement operator. Here we consider parallel imaging problem in MRI where
 the A operator consists of undersampling mask, FFT, and coil sensitivity maps.

Dw(x): it represents the residual learning CNN.

Here is the description of the parameters that you can modify below.

epochs: how many times to pass through the entire dataset

nLayer: number of layers of the convolutional neural network.
        Each layer will have filters of size 3x3. There will be 64 such filters
        Except at the first and the last layer.

gradientMethod: MG or AG. set MG for 'manual gradient' of conjuagate gradient (CG) block
                as discussed in section 3 of the above paper. Set it to AG if
                you want to rely on the tensorflow to calculate gradient of CG.

K: it represents the number of iterations of the alternating strategy as
    described in Eq. 10 in the paper.  Also please see Fig. 1 in the above paper.
    Higher value will require a lot of GPU memory. Set the maximum value to 20
    for a GPU with 16 GB memory. Higher the value more is the time required in training.

sigma: the standard deviation of Gaussian noise to be added in the k-space

batchSize: You can reduce the batch size to 1 if the model does not fit on GPU.

Output:

After running the code the output model will be saved in the subdirectory 'savedModels'.
You can give the name of the generated ouput directory in the tstDemo.py to
run the newly trained model on the test data.


@author: Hemant Kumar Aggarwal
"""

# import some librariesw
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import supportingFunctions as sf
import model as mm

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

#--------------------------------------------------------------
#% SET THESE PARAMETERS CAREFULLY
nLayers=5
epochs=50
batchSize=1
gradientMethod='AG'
K=1
sigma=0.01
restoreWeights=False
#%% to train the model with higher K values  (K>1) such as K=5 or 10,
# it is better to initialize with a pre-trained model with K=1.
if K>1:
    restoreWeights=True
    restoreFromModel='04Jun_0243pm_5L_1K_100E_AG'

if restoreWeights:
    wts=sf.getWeights('savedModels/'+restoreFromModel)
#--------------------------------------------------------------------------
#%%Generate a meaningful filename to save the trainined models for testing
print ('*************************************************')
start_time=time.time()
saveDir='savedModels/'
cwd=os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b_%I%M%P_")+ \
 str(nLayers)+'L_'+str(K)+'K_'+str(epochs)+'E_'+gradientMethod

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName= directory+'/model'


#%% save test model
tf.reset_default_graph()

csmT = tf.placeholder(tf.complex64,shape=(None,12,256,232),name='csm')
maskT= tf.placeholder(tf.complex64,shape=(None,256,232),name='mask')
atbT = tf.placeholder(tf.float32,shape=(None,256,232,2),name='atb')

out=mm.makeModel(atbT,csmT,maskT,False,nLayers,K,gradientMethod)
predTst=out['dc'+str(K)]
predTst=tf.identity(predTst,name='predTst')
sessFileNameTst=directory+'/modelTst'

saver=tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    savedFile=saver.save(sess, sessFileNameTst,latest_filename='checkpointTst')
print ('testing model saved:' +savedFile)
#%% read multi-channel dataset
trnOrg,trnAtb,trnCsm,trnMask=sf.getData('training')
trnOrg,trnAtb=sf.c2r(trnOrg),sf.c2r(trnAtb)

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

trnData = tf.data.Dataset.from_tensor_slices((orgP,atbP,csmP,maskP))
trnData = trnData.cache()
trnData=trnData.repeat(count=epochs)
trnData = trnData.shuffle(buffer_size=trnOrg.shape[0])
trnData=trnData.batch(batchSize)
trnData=trnData.prefetch(5)
iterator=trnData.make_initializable_iterator()
orgT,atbT,csmT,maskT = iterator.get_next('getNext')

#%% make training model

out=mm.makeModel(atbT,csmT,maskT,True,nLayers,K,gradientMethod)
predT=out['dc'+str(K)]
predT=tf.identity(predT,name='pred')
loss = tf.reduce_mean(tf.reduce_sum(tf.pow(predT-orgT, 2),axis=0))
tf.summary.scalar('loss', loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer()
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    opToRun=optimizer.apply_gradients(capped_gvs)


#%% training code


print ('training started at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('parameters are: Epochs:',epochs,' BS:',batchSize,'nSteps:',nSteps,'nSamples:',nTrn)

saver = tf.train.Saver(max_to_keep=100)
totalLoss,ep=[],0
lossT = tf.placeholder(tf.float32)
lossSumT = tf.summary.scalar("TrnLoss", lossT)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    if restoreWeights:
        sess=sf.assignWts(sess,nLayers,wts)

    feedDict={orgP:trnOrg,atbP:trnAtb, maskP:trnMask,csmP:trnCsm}
    sess.run(iterator.initializer,feed_dict=feedDict)
    savedFile=saver.save(sess, sessFileName)
    print("Model meta graph saved in::%s" % savedFile)

    writer = tf.summary.FileWriter(directory, sess.graph)
    for step in tqdm(range(nSteps)):
        try:
            tmp,_,_=sess.run([loss,update_ops,opToRun])
            totalLoss.append(tmp)
            if np.remainder(step+1,nBatch)==0:
                ep=ep+1
                avgTrnLoss=np.mean(totalLoss)
                lossSum=sess.run(lossSumT,feed_dict={lossT:avgTrnLoss})
                writer.add_summary(lossSum,ep)
                totalLoss=[] #after each epoch empty the list of total loos
        except tf.errors.OutOfRangeError:
            break
    savedfile=saver.save(sess, sessFileName,global_step=ep,write_meta_graph=True)
    writer.close()

end_time = time.time()
print ('Trianing completed in minutes ', ((end_time - start_time) / 60))
print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('*************************************************')

#%%

