import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn.functional as F
import numpy as np
# Device configuration
import sys

import matplotlib.pyplot as plt

def mnist_py_data(batch_size):
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # print(train_dataset,'tr')
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader,test_loader

def preprocess(x, y):
    # [b, 28, 28], [b]
    # print(x.shape, y.shape)


    x=torch.from_numpy(x)
    x=x.type(torch.float32)/255
    x = torch.reshape(x, (-1, 28 * 28))

    y=torch.from_numpy(y)
    y=torch.nn.functional.one_hot(y.to(torch.int64))
    return x.numpy(), y.numpy()


def preprocess_rnn(x, y):
    # [b, 28, 28], [b]
    # print(x.shape, y.shape)
    x = torch.from_numpy(x)
    x = x.type(torch.float32) / 255


    y = torch.from_numpy(y)

    y = torch.nn.functional.one_hot(y.to(torch.int64))

    return x.numpy(), y.numpy()


def open_file(filename):
    info = []
    data = []
    count = 0
    with open(filename,'r') as f:
        for line in f.readlines():
            # first 6 lines has information
            # assuming its the same for all files
            if count < 6:
                info.append(line)
                count += 1
                continue
            line = line.strip().split(',')
            data.append(line)
    return "".join(info), data

def LSTMcell_cost(input_size,hidden_size,batch_size=1):
   LSTM_w1=np.random.random((input_size,hidden_size))
   LSTM_w1_c = 3*sys.getsizeof(LSTM_w1)
   LSTM_w2=np.random.random((hidden_size,hidden_size))
   LSTM_w2_c=3*sys.getsizeof(LSTM_w2)
   LSTM_b1=np.random.random((batch_size,hidden_size))
   LSTM_b1_c =4*sys.getsizeof(LSTM_b1)
   LSTM_cost=LSTM_w1_c+LSTM_w2_c+LSTM_b1_c
   return LSTM_cost

def Linear_cost(input_size,hidden_size,batch_size=1):
    Linear_w=np.random.random((input_size,hidden_size))
    Linear_w_c= sys.getsizeof(Linear_w)
    Linear_b=np.random.random((hidden_size,batch_size))
    Linear_b_c= sys.getsizeof(Linear_b)
    Linear_total_cost=Linear_b_c+Linear_w_c
    return Linear_total_cost

def PMFparameter(model0):

    L1 = model0.linear.weight
    LS1h = model0.lstm1.weight_hh
    LS1i = model0.lstm1.weight_ih
    LS2h = model0.lstm2.weight_hh
    LS2i = model0.lstm2.weight_ih
    LS3h = model0.lstm3.weight_hh
    LS3i = model0.lstm3.weight_ih
    L2 = model0.linear1.weight

    return L1,LS1h,LS1i,LS2h,LS2i,LS3h,LS3i,L2

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def final_acc(accs_tre,acc):
    unit = np.linspace(0, 4, num=1200)
    unit = sigmoid(unit)
    unit2 = np.linspace(0, 5.5, num=1200)
    unit2 = sigmoid(unit2)
    unit3 = np.linspace(0, 7, num=1200)
    unit3 = sigmoid(unit3)
    ny_ratio=0.50
    accs_tre=np.array(accs_tre)/100
    acc=np.array(acc)/100
    acc=(accs_tre*acc*ny_ratio)*100
    return acc

def client_energy(time):
    occupation_rate= 0.11
    standard_power=45
    energy=time*occupation_rate*standard_power

    return energy


def client_time(time):
    Linear_Training_time_sample = 5.803506851196289
    total_training_time_sample = 44.50100803375244
    ratio = Linear_Training_time_sample / total_training_time_sample
    client_time=time*ratio
    return client_time
def batch_split(x,batch_size):
    x1=np.split(x,batch_size)
    return x1

def data_process(data):
    data = np.array(data)
##116.65
###40.1
    la = data[:, 1].astype(np.float) / 116.65
    lo = data[:, 0].astype(np.float) / 40.1


    la = la.reshape(-1, 1)
    lo = lo.reshape(-1, 1)
    lalo = np.hstack((la, lo))
    return  la,lo,lalo

def data_simplify(lalo):
    location=[]
    for i in range(280):
        location.append(lalo[i*50])
        loca = np.array(location)
    return loca

def place_count(la,lalo):
    place=la
    for i ,array in enumerate(lalo):

        if 0.9965<lalo[i,0]<0.9970 and 0.997<lalo[i,1]<0.998:
            place[i]=12
        if 0.9970<lalo[i,0]<0.9975 and 0.997<lalo[i,1]<0.998:
            place[i]=1
        if 0.9975 < lalo[i, 0] < 0.998 and 0.997 < lalo[i, 1] < 0.998:
            place[i] = 2
        if 0.998 < lalo[i, 0] < 0.9985 and 0.997 < lalo[i, 1] < 0.998:
            place[i] = 3
        if 0.998 < lalo[i, 0] < 0.9985 and 0.996 < lalo[i, 1] < 0.997:
            place[i] = 8
        if 0.9985 < lalo[i, 0] < 0.999 and 0.997 < lalo[i, 1] < 0.998:
            place[i] = 4
        if 0.9985 < lalo[i, 0] < 0.999 and 0.998 < lalo[i, 1] < 0.999:
            place[i] = 14
        if 0.999 < lalo[i, 0] < 0.9995 and 0.998 < lalo[i, 1] < 0.999:
            place[i] = 5
        if 0.999 < lalo[i, 0] < 0.9995 and 0.999 < lalo[i, 1] :
            place[i] = 6
        if 0.9995 < lalo[i, 0] < 1 and 0.998 < lalo[i, 1] < 0.999:
            place[i] = 7
        if 0.9975 < lalo[i, 0] < 0.998 and 0.994 < lalo[i, 1] < 0.996:
            place[i] = 9
        if 0.9970 < lalo[i, 0] < 0.9975 and 0.994 < lalo[i, 1] < 0.996:
            place[i] = 10
        if 0.9970 < lalo[i, 0] < 0.9975 and 0.996 < lalo[i, 1] < 0.997:
            place[i] = 11
        if 0.9975 < lalo[i, 0] < 0.998 and 0.996 < lalo[i, 1] < 0.997:
            place[i] = 13
        place=np.reshape(place,-1)
    return place
def route_extend(pla):
    place=pla

    for i in range(1000):
        place=np.append(place,pla)
    return place
def data_y_divide(x,sequence_length):
    x_d = [[x[j:sequence_length + j]] for j in range(len(x) - sequence_length)]

    y_t = [x[i + sequence_length] for i in range(len(x) - sequence_length)]

    return x_d,y_t

def one_hot_convert(data_y_divide_train):
    data_y_divide_train = torch.from_numpy(np.array(data_y_divide_train))
    labels = torch.nn.functional.one_hot(data_y_divide_train.to(torch.int64))
    return labels

def data_general(x,sequence_length):
    places = route_extend(x)

    places = torch.from_numpy(np.array(places))
    places = torch.nn.functional.one_hot(places.to(torch.int64))
    places=places.numpy()
    x_d = [[places[j:sequence_length + j]] for j in range(len(places) - sequence_length)]
    y_t = [places[i + sequence_length] for i in range(len(places) - sequence_length)]
    data_x_divide_train=np.array(x_d)
    data_y_divide_train=np.array(y_t)
    data_x_divide_train=np.squeeze(data_x_divide_train)
    return data_x_divide_train,data_y_divide_train

def batch_creat(x,batch_size):

    x_tt = np.split(x, batch_size)
    x_tt = np.array(x_tt)
    return x_tt

def experiment():
    info, data = open_file('20090403011657.plt')
    la,lo,lalo=data_process(data)
    location=data_simplify(lalo)
    la=data_simplify(la)

    pla=place_count(la,location)
    places=route_extend(pla)
    #print((places/14)[0:40])
    abandon_x,data_y_divide_train=data_y_divide(places, 28)
    data_x_divide_train,abandon_y=data_y_divide(places/14, 28)
    #print(data_x_divide_train)
    #print(data_x_divide_train[0])
    #print(data_y_divide_train[0])
    print(data_y_divide_train[1:5])
    data_y_divide_train=torch.from_numpy(np.array(data_y_divide_train))
    labels = torch.nn.functional.one_hot(data_y_divide_train.to(torch.int64))
#data_x_divide_train=data_x_divide_train/14
    print(data_x_divide_train[1:5])
    print(labels[0:5])

    #print(places,'pla')
    #plt.plot(location[:,0],location[:,1])
    #plt.plot(location[25:27,0],location[25:27,1])
    plt.plot(places[0:400])
    plt.grid()
    plt.show()

    pla=[ 1,  1,  2,  2,  2,3,  3,  8,  4,  5,  7,  7,  7,  6,  6,  5,  5,  4,
      8,  3,  2,  2,  2,  1,
      1,  1,  1,  1,  1,  1,  1, 12, 12,12, 12, 12, 12, 12, 12, 1, 11, 11, 11, 9,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10,
     10, 11, 11, 11, 11, 11,  1,  1,  1,  1]


