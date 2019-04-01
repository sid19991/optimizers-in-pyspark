from __future__ import print_function

import re
import sys
import numpy as np
from operator import add
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import math
def f(partition,theta,m):
    for sample in partition:
        j=1;
        sum1=theta[0];
        at=[theta[0]]
        for i in sample:
            if(j<len(theta)):
                sum1+=i*theta[j]
            else:
                sum1-=i;
            j+=1
        #print('grad: '+str(sum1))
        #print((sum1*sample)/m)
        yield((sum1*sample)/m);
def f1(partition,theta,m):
    for sample in partition:
        j=1;
        sum1=theta[0];
        for i in sample:
            if(j<len(theta)):
                sum1+=i*theta[j]
            else:
                sum1-=i;
            j+=1
        #print('loss: '+str(sum1))
        yield((sum1*sum1)/(2*m))
def parallelSGD(df,iter=1000):
	#Parelled SGD
	theta=np.zeros((len(df.columns),1));
	den=df.count();
	progress=[]
	for i in range(iter):
	    tmp=df.rdd.mapPartitions(lambda partition: f1(partition,theta,den)).treeReduce(add);
	    grads=df.rdd.mapPartitions(lambda partition:f(partition,theta,den)).treeReduce(add);
	    #print(grads)
	    for j in range(len(theta)):
		theta[j]=theta[j]-0.000001*grads[j]
	    progress.append(tmp)
	    print(tmp)
	    print(i)
	plt.figure()
	plt.title('Loss function in paralled SGD')
	plt.plot(progress)
def Adam(df,iter=1000):
	#Adam
	theta=np.zeros((len(df.columns),1));
	den=df.count();
	alpha=0.001
	beta_1=0.9;
	beta_2=0.999;
	eps=1e-8;
	m_t=np.zeros((len(df.columns),1),dtype=float);
	v_t=np.zeros((len(df.columns),1),dtype=float);
	progress=[]
	for i in range(iter):
	    tmp=df.rdd.mapPartitions(lambda partition: f1(partition,theta,den)).treeReduce(add);
	    grads=df.rdd.mapPartitions(lambda partition:f(partition,theta,den)).treeReduce(add);
	    grads=grads.reshape(len(df.columns),1)
	    m_t=beta_1*m_t+(1-beta_1)*grads;
	    v_t=(beta_2)*v_t+(1-beta_2)*(grads*grads);
	    mcap_t=m_t/(1-math.pow(beta_1,float(i+1)));
	    vcap_t=v_t/(1-math.pow(beta_2,float(i+1)));
	    progress.append(tmp)
	    #print(mcap_t.shape)
	    print(tmp)
	    for j in range(len(theta)):
		tmp=alpha*(mcap_t[j]/(math.sqrt(vcap_t[j])+eps));
		theta[j]=theta[j]-tmp
	    
	    
	    print(i)
	plt.figure()
	plt.title('Loss function in Adam')
	plt.plot(progress)
def AdaGrad(df,iter=1000):
	#AdaGrad
	theta=np.zeros((len(df.columns),1));
	den=df.count();
	eta=0.9
	eps=1e-8;
	Gt=np.zeros((1,len(df.columns)));
	progress=[]
	for i in range(iter):
	    tmp=df.rdd.mapPartitions(lambda partition: f1(partition,theta,den)).treeReduce(add);
	    grads=df.rdd.mapPartitions(lambda partition:f(partition,theta,den)).treeReduce(add);
	    grads=grads.reshape(len(df.columns),1)
	    Gt=Gt+np.sum((grads*grads),axis=0);
	    progress.append(tmp)
	    #print(mcap_t.shape)
	    print(tmp)
	    for j in range(len(theta)):
		tmp=(eta*grads[j])/(math.sqrt(Gt[0][j]+eps));
		theta[j]=theta[j]-tmp;
	    
	    
	    print(i)
	plt.figure()
	plt.title('Loss function in AdaGrad')
	plt.plot(progress)
def AdaDelta(df,iter=1000):
	#AdaDelta
	theta=np.zeros((len(df.columns),1));
	den=df.count();
	rho=0.95
	eps=1e-6;
	eta=0.009
	EGt=np.zeros((1,len(df.columns)));
	progress=[]
	for i in range(iter):
	    tmp=df.rdd.mapPartitions(lambda partition: f1(partition,theta,den)).treeReduce(add);
	    grads=df.rdd.mapPartitions(lambda partition:f(partition,theta,den)).treeReduce(add);
	    grads=grads.reshape(len(df.columns),1)
	    EGt=rho*EGt+(1-rho)*np.sum((grads*grads),axis=0);
	    progress.append(tmp)
	    #print(mcap_t.shape)
	    print(tmp)
	    for j in range(len(theta)):
		tmp=(eta*grads[j])/(math.sqrt(EGt[0][j]+eps));
		theta[j]=theta[j]-tmp;
	    
	    
	    print(i)
	plt.figure()
	plt.title('Loss function in AdaDelta')
	plt.plot(progress)
def AdaMax(df,iter=1000):
	#AdaMax
	theta=np.zeros((len(df.columns),1));
	den=df.count();
	alpha=0.00001
	beta_1=0.9;
	beta_2=0.999;
	eps=1e-8;
	m_t=np.zeros((len(df.columns),1),dtype=float);
	v_t=np.zeros((len(df.columns),1),dtype=float);
	progress=[]
	for i in range(iter):
	    tmp=df.rdd.mapPartitions(lambda partition: f1(partition,theta,den)).treeReduce(add);
	    grads=df.rdd.mapPartitions(lambda partition:f(partition,theta,den)).treeReduce(add);
	    grads=grads.reshape(len(df.columns),1)
	    m_t=beta_1*m_t+(1-beta_1)*grads;
	    v_t=np.maximum((beta_2)*v_t,(1-beta_2)*np.abs(grads));
		
	    progress.append(tmp)
	    #print(mcap_t.shape)
	    print(tmp)
	    for j in range(len(theta)):
		tmp=(alpha/(1-math.pow(beta_1,i+1)))*(m_t[j]/(v_t[j]));
		theta[j]=theta[j]-tmp;
	    
	    
	    print(i)
	plt.figure()
	plt.title('Loss function in AdaMax')
	plt.plot(progress)

spark = SparkSession\
      .builder\
        .appName("optimizers")\
        .getOrCreate()
df = spark.read.csv("winequality-red.csv",sep=';',header=True);
for i in df.columns:
    df = df.withColumn(i, df[i].cast('float'))
parallelSGD(df);
Adam(df);
AdaGrad(df);
AdaDelta(df);
AdaMax(df);
plt.show()
plt.show();
