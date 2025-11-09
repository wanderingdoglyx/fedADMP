
import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd




def differential_privacy_single_1d(w,epsilon=40,client_number=1,p=0):
    #w1 = w[:]
    if p==0:
        return w

    cut1=len(w)*p
    cut=round(cut1)
    w1= w[:cut]
    #print(w1,'w1')
    row = len(w1)
#    w2=w2.numpy()
    delta=w
    theta = 10e-5
    s=np.linalg.norm(delta)
   # print(s)
   # print(s)
   # print('------------------------------',s)
    sigma = ((s * (np.sqrt(2 * (np.log(1.25 / theta))))) / epsilon)
    noise = np.random.normal(0, sigma, [row])
    noise=((noise) / client_number)

    ni= np.pad(noise,(0,len(w)-len(noise)),'constant',constant_values=(0,0))
    ni=ni.reshape((-1,1))
    #print(ni)

    return ni


def differential_privacy_pre_2d(w,epsilon=40,client_number=1,p=0):
    # w1 = w[:]
    if p == 0:
        return w

    row = w.shape[0]
    column = w.shape[1]

    theta = 10e-5
    cut1 = row*column* p
    cut = round(cut1)
    w1 = w[:cut]
    # print(w1,'w1')

    delta = w

    s = np.linalg.norm(delta)
    # print(s)
    # print(s)
    # print('------------------------------',s)
    # print(ni)
    sigma = ((s * (np.sqrt(2 * (np.log(1.25 / theta))))) / epsilon)

    noise = np.random.normal(0, sigma, [row, column])
    #  noise = 0  #############################################################################################################
    noise = ((noise) / client_number)
    #w_delta = w1 + ((noise) / client_number)

    return noise

def differential_privacy_pre(w1,w2):
    #w1 = w[:]

#    w2=w2.numpy()
    delta=(w1 - w2)
    #print('-----------------',delta)
    #print('delta',delta)
    delta1=tf.Variable(delta)
    #print(delta1[1,1])
    #print('delta', delta1)
   # delta3=tf.Variable(delta)
    s=np.linalg.norm(delta)
   # print('------------------------------',s)

    return s,delta1




def differential_privacy_mat(w1,client_number,s,epsilon):

    row = w1.shape[0]
    column = w1.shape[1]


    theta = 10e-5

    sigma = ((s* (np.sqrt(2 * (np.log(1.25 / theta))))) / epsilon)

    noise = np.random.normal(0, sigma, [row, column])
  #  noise = 0  #############################################################################################################

    w_delta = w1+((noise ) / client_number)

    return (tf.Variable(w_delta))


