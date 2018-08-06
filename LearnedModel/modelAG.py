"""
Created on Mon May 7th, 2018 at 1:30 PM

@author: haggarwal
"""

import tensorflow as tf
import numpy as np
from os.path import expanduser
home = expanduser("~")
epsilon=1e-5
TFeps=tf.constant(1e-5,dtype=tf.float32)

c2r=lambda x:tf.stack([tf.real(x),tf.imag(x)],axis=-1)
r2c=lambda x:tf.complex(x[...,0],x[...,1])

def myConv2d(x, szW, szB,trainning,bnRelu, strides=1):
    W=tf.get_variable('W',shape=szW,initializer=tf.contrib.layers.xavier_initializer())
    #b=tf.get_variable('b',shape=szB,initializer=tf.contrib.layers.xavier_initializer())
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    #xb = tf.nn.bias_add(x, b)
    xbn=tf.layers.batch_normalization(x,training=trainning,fused=True,name='BN')
    if bnRelu:
        return tf.nn.relu(xbn)
    else:
        return xbn

def funDw(inp,trainning,nLay):
    bnRelu=True
    nw={}
    nw['c'+str(0)]=inp
    szW,szB={},{}
    szW = {key: (3,3,64,64) for key in range(2,nLay)}
    szW[1]=(3,3,2,64)
    szW[nLay]=(3,3,64,2)
    szB = {key: (64,) for key in range(1,nLay)}
    szB[nLay]=(2,)

    for i in np.arange(1,nLay+1):
        if i==nLay:
            bnRelu=False
        with tf.variable_scope('Layer'+str(i)):
            nw['c'+str(i)]=myConv2d(nw['c'+str(i-1)],szW[i],szB[i],trainning,bnRelu)

    with tf.name_scope('Residual'):
        shortcut=tf.identity(inp)
        dw=shortcut+nw['c'+str(nLay)]
    return nw,dw
class Aclass:
    def __init__(self, csm,mask,lam):
        with tf.name_scope('Ainit'):
            s=tf.shape(mask)
            self.nrow,self.ncol=s[0],s[1]
            self.pixels=self.nrow*self.ncol
            self.mask=mask
            self.csm=csm
            self.SF=tf.complex(tf.sqrt(tf.to_float(self.pixels) ),0.)
            self.lam=lam
    def myAtA(self,img):
        with tf.name_scope('AtA'):
            coilImages=self.csm*img
            kspace=  tf.fft2d(coilImages)/self.SF
            temp=kspace*self.mask
            coilImgs =tf.ifft2d(temp)*self.SF
            coilComb= tf.reduce_sum(coilImgs*tf.conj(self.csm),axis=0)
            coilComb=coilComb+self.lam*img
        return coilComb
def myCG(A,rhs,maxIter=5,tol=1e-15):

    cond=lambda i,rTr,*_: tf.logical_and( tf.less(i,maxIter), rTr>tol)
    def body(i,rTr,x,r,p):
        with tf.name_scope('cgBody'):
            Ap=A.myAtA(p)
            alpha = rTr / tf.to_float(tf.reduce_sum(tf.conj(p)*Ap))
            alpha=tf.complex(alpha,0.)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = tf.to_float( tf.reduce_sum(tf.conj(r)*r))
            beta = rTrNew / rTr
            beta=tf.complex(beta,0.)
            p = r + beta * p
        return i+1,rTrNew,x,r,p

    x=tf.zeros_like(rhs)
    i,r,p=0,rhs,rhs
    rTr = tf.to_float( tf.reduce_sum(tf.conj(r)*r),)
    loopVar=i,rTr,x,r,p
    out=tf.while_loop(cond,body,loopVar,name='CGwhile')[2]
    return out


def funDC(dwX,atb,csm,mask,cgIter):
    lam1=tf.get_variable('lam1', shape=(1,), initializer=tf.zeros_initializer())
    #lam1=tf.get_variable('lam1',  initializer=np.asarray([.005]).astype(np.float32))
    rhs = atb+ lam1*dwX
    #rhs = tf.complex(xsum[...,0],xsum[...,1])
    #rhs=c2r(rhs)
    lam2=tf.complex(lam1,0.)

    def fn( (csm,mask,rhs) ):
        Aobj=Aclass( csm,mask,lam2 )
        def f(x):
            x=r2c(x)
            y=myCG(Aobj,x,maxIter=cgIter)
            y=c2r(y)
            return y
        y=f(rhs)
        return y
    inp=(csm,mask,rhs)
    rec=tf.map_fn(fn,inp,dtype=tf.float32,name='mapFn' )
    #rec=r2c(rhs)


    #recRealImag=tf.stack([tf.real(rec),tf.imag(rec)],axis=-1)
    return rec#recRealImag


def makeModel(atb,csm,mask,trainning,nLayers,nRepetitions,nCoils,wtSharing,cgIter):
    with tf.name_scope('myModel'):
        nw={}
        out={}
        #with tf.device("/gpu:0"):
        with tf.variable_scope('Wts'):
            nw[1],out['dw1']=funDw(atb,trainning,nLayers)
            out['dc1']=funDC(out['dw1'],atb,csm,mask,cgIter)

        for i in np.arange(2,nRepetitions+1):
            j=str(i)
            if wtSharing:
                with tf.variable_scope('Wts',reuse=True):
                    nw[i],out['dw'+j]=funDw(out['dc'+str(i-1)],trainning,nLayers)
                    out['dc' +j]=funDC(out['dw'+j],atb,csm,mask,cgIter)
            else:
                with tf.variable_scope('Wts'+j):
                    nw[i],out['dw'+j]=funDw(out['dc'+str(i-1)],trainning,nLayers)
                    out['dc' +j]=funDC(out['dw'+j],atb,csm,mask,cgIter)
    return nw,out

def makeModelTst(inpImg,atb,csm,mask,nLayers,cgIter):
    with tf.name_scope('myModel'):
        with tf.variable_scope('Wts'):
            nw,dw=funDw(inpImg,False,nLayers)
            dc=funDC(dw,atb,csm,mask,cgIter)

    return nw,dw,dc