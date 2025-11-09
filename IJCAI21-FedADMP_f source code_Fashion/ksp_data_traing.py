import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import mnist_numpy
import differential_privacy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###############################################################################
def KSP_training_time_energy():
    from function_ksp import mnist_py_data,open_file,batch_split,data_process,data_y_divide,data_simplify,place_count,route_extend,\
        one_hot_convert,data_general,preprocess,preprocess_rnn,batch_creat,final_acc,client_energy,client_time,Linear_cost
    num_classes = 10+19###############################################
    num_epochs = 2  ############################  #2#  ###################
    batch_size = 100
    learning_rate = 0.001

    input_size = 28+19#############################################
    sequence_length = 28
    hidden_size = 128#############################################
    num_layers = 1
    output_size=num_classes

    ####################################################################################################
    client_number=1
    class Sequence(nn.Module):
        def __init__(self,input_size,output_size,hidden_size):
            super(Sequence, self).__init__()
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.lstm1 = nn.LSTMCell(hidden_size, hidden_size)
            self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
            self.lstm3 = nn.LSTMCell(hidden_size, hidden_size)
            self.linear = nn.Linear(hidden_size, output_size)

        def forward(self, input):

         #   h_t = torch.zeros(input.size(0), hidden_size, dtype=torch.float)
         #   c_t = torch.zeros(input.size(0), hidden_size, dtype=torch.float)
          #  h_t2 = torch.zeros(input.size(0), hidden_size, dtype=torch.float)
         #   c_t2 = torch.zeros(input.size(0), hidden_size, dtype=torch.float)

            h_t = torch.zeros(batch_size,hidden_size, dtype=torch.float).to(device)
            c_t = torch.zeros(batch_size,hidden_size, dtype=torch.float).to(device)
            h_t2 = torch.zeros(batch_size,hidden_size, dtype=torch.float).to(device)
            c_t2 = torch.zeros(batch_size,hidden_size, dtype=torch.float).to(device)

            h_t3 = torch.zeros(batch_size, hidden_size, dtype=torch.float).to(device)
            c_t3 = torch.zeros(batch_size, hidden_size, dtype=torch.float).to(device)


            for input_t in input.split(1, dim=1):


                #print(input_t.size(),'t')
               # print(input_t)

                input_t=input_t.view(-1, input_size)#############################################
               # print(input_t.size(), 't')
               # print(input_t.size(), 't')
               # print(input_t)
                input_x= self.linear1(input_t)
                ni = differential_privacy.differential_privacy_pre_2d(input_x.cpu().detach().numpy(), epsilon=100,
                                                                      client_number=3, p=1)
                # print(ni,'ni')#########################################################################
                input_x = input_x + torch.from_numpy(ni).float().to(device)

                #print(input_x.size(),'x')
               # print(input_x,'v')
                h_t, c_t = self.lstm1(input_x, (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear(torch.sigmoid(h_t3))
                #outputs += [output]
            #for i in range(future):# if we should predict the future
               # h_t, c_t = self.lstm1(output, (h_t, c_t))
                #h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                #output = self.linear(h_t2)
              #  outputs += [output]
            #outputs = torch.cat(outputs, dim=1)
            #print(output,'output')###############################################################################
            #print(output.size())################################################################
            #output=nn.functional.relu(output)
            return output


    pla=[ 7, 7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
      8,  8,  8,  8,  8,  8, 7,  7,  7,  7,  7, 7,  7,  7,  7,  7,  7,  6,
      6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  6,  6,
      9,  9,  9,  9,  9, 12, 12, 12, 12, 14, 14, 15, 15, 16, 16, 17, 17, 18,
     18, 18, 18, 18, 18, 17, 17, 16, 16, 15, 15, 15, 14, 14, 14, 14, 14, 14,
     14, 12, 12, 12, 13, 10, 10,  7,  7,  7,  4,  2,  2,  3,  2,  1,  1,  1,
      2,  3,  5,  5,  5,  8,  8, 7, 7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
      8,  8,  8,  8,  8,  8, 7,  7,  7,  7,  7, 7,  7,  7,  7,  7,  7,  6,
      6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  6,  6,
      9,  9,  9,  9,  9, 12, 12, 12, 12, 14, 14, 15, 15, 16, 16, 17, 17, 18,
     18, 18, 18, 18, 18, 17, 17, 16, 16, 15, 15, 15, 14, 14, 14, 14, 14, 14,
     14, 12, 12, 12, 13, 10, 10,  7,  7,  7,  4,  2,  2,  3,  2,  1,  1,  1,
      2,  3,  5,  5,  5,  8,  8]



    x1, y1, x2, y2=mnist_numpy.load()
    x1,y1=preprocess_rnn(x1,y1)
    x2,y2=preprocess_rnn(x2,y2)
    x2=x2[0:6000]
    y2=y2[0:6000]
    x1=x1.reshape(x1.shape[0],28,28)
    x2=x2.reshape(x2.shape[0],28,28)

    data_x_divide_train,data_y_divide_train=data_general(pla,sequence_length)
    data_x_test=data_x_divide_train[0:6000]
    data_y_test=data_y_divide_train[0:6000]

    data_x_divide_train=data_x_divide_train[0:x1.shape[0]]
    data_y_divide_train=data_y_divide_train[0:y1.shape[0]]
    z = np.concatenate((x1, data_x_divide_train), axis=2)
    zy= np.concatenate((y1, data_y_divide_train), axis=1)
    z_test=np.concatenate((x2, data_x_test), axis=2)
    zy_test=np.concatenate((y2, data_y_test), axis=1)


    x1=batch_creat(x1,x1.shape[0]/batch_size)
    y1=batch_creat(y1,y1.shape[0]/batch_size)
    x2=batch_creat(x2,x2.shape[0]/batch_size)
    y2=batch_creat(y2,y2.shape[0]/batch_size)
    data_x_divide_train=batch_creat(data_x_divide_train,data_x_divide_train.shape[0]/batch_size)
    data_y_divide_train=batch_creat(data_y_divide_train,data_y_divide_train.shape[0]/batch_size)
    data_x_test=batch_creat(data_x_test,data_x_test.shape[0]/batch_size)
    data_y_test=batch_creat(data_y_test,data_y_test.shape[0]/batch_size)

    z=batch_creat(z, z.shape[0]/batch_size)
    zy=batch_creat(zy, zy.shape[0]/batch_size)
    z_test=batch_creat(z_test, z_test.shape[0]/batch_size)
    zy_test=batch_creat(zy_test, zy_test.shape[0]/batch_size)

    model = Sequence(input_size,output_size,hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps = len(z)
    start_time = time.time()
    accs = []
    accs_tre=[]
    #print(x1.shape[0])
    iters=x1.shape[0]
    #print(iters,'iter')
    for epoch in range(num_epochs):
        for i in range(iters):
           # print(i,'i')
            # origin shape: [N, 1, 28, 28]
            a = np.zeros((model.linear1.weight.cpu().detach().numpy().shape))

            client = 0
            if client < client_number:
                z_input=z[i]
                zy_input=zy[i]
                z_input=torch.from_numpy(z_input)
                zy_input=torch.from_numpy(zy_input)
                z_input=z_input.to(device)
                zy_input=zy_input.to(device)

                #print(images[0],'image')
                #print(images,'image')
                #print(images.size())


               # print(labels,'label')
               # print(labels.size())
                # Forward pass
                outputs = model(z_input.float())
                loss = criterion(outputs,zy_input.float())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    a = model.linear1.weight.cpu().numpy() / client_number
                client = client + 1

            with torch.no_grad():


                model.linear1.weight = nn.Parameter(torch.from_numpy(a).cuda().float())

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

    #training_time = time.time() - start_time
        # Test the model
                            # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():


            n_correct = 0
            n_samples = 0
            n_correct1=0
            n_samples1=0
            for i in range(z_test.shape[0]):

                input_test=z_test[i]
                label_test=zy_test[i]

                input_test=torch.from_numpy(input_test)
                label_test=torch.from_numpy(label_test)

                input_test=input_test.to(device)
                label_test=label_test.to(device)

                label_test_image=label_test[:,0:10]#########################################
                label_test_tre=label_test[:,10:]##############################################

                outputs = model(input_test.float())

                # max returns (value ,index)
                outputs_image=outputs[:,0:10]
                outputs_tre=outputs[:,10:]

                _, predicted = torch.max(outputs_image.data, 1)
                _, label_test_image_convert = torch.max(label_test_image.data, 1)
                _, predicted_tre = torch.max(outputs_tre.data, 1)
                _, label_test_tre_convert = torch.max(label_test_tre.data, 1)
               # print(predicted,'pre')
               # print(label_test,'label')

                n_samples += label_test_image.size(0)
                n_correct += (predicted == label_test_image_convert).sum().item()
                n_samples1 += label_test_tre.size(0)
                n_correct1 += (predicted_tre == label_test_tre_convert).sum().item()


            acc = 100.0 * n_correct / n_samples
            accs.append(acc)

            acc_tre=100.0 * n_correct1 / n_samples1

            accs_tre.append(acc_tre)

    training_time = time.time() - start_time
    facc = final_acc(accs_tre, acc)
    print(facc,'FedADMP')
    ct = client_time(training_time)
    e = client_energy(ct)
    com = num_epochs * iters * (Linear_cost(input_size, hidden_size, batch_size))
    print(com, 'communication cost')
    print(ct, 'client time')
    print(e, 'energy')

#return ct, e
        # print(accs,'acc')
    # print(f'Accuracy of the network on the 10000 test images: {acc} %')
def KSP_training_acc():
    from function_ksp import mnist_py_data,open_file,batch_split,data_process,data_y_divide,data_simplify,place_count,route_extend,\
        one_hot_convert,data_general,preprocess,preprocess_rnn,batch_creat,final_acc,client_energy,client_time
    num_classes = 10+19###############################################
    num_epochs = 2  ############################  #2#  ###################
    batch_size = 100
    learning_rate = 0.001

    input_size = 28+19#############################################
    sequence_length = 28
    hidden_size = 128#############################################
    num_layers = 1
    output_size=num_classes

    ####################################################################################################
    client_number=1
    class Sequence(nn.Module):
        def __init__(self,input_size,output_size,hidden_size):
            super(Sequence, self).__init__()
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.lstm1 = nn.LSTMCell(hidden_size, hidden_size)
            self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
            self.lstm3 = nn.LSTMCell(hidden_size, hidden_size)
            self.linear = nn.Linear(hidden_size, output_size)

        def forward(self, input):

         #   h_t = torch.zeros(input.size(0), hidden_size, dtype=torch.float)
         #   c_t = torch.zeros(input.size(0), hidden_size, dtype=torch.float)
          #  h_t2 = torch.zeros(input.size(0), hidden_size, dtype=torch.float)
         #   c_t2 = torch.zeros(input.size(0), hidden_size, dtype=torch.float)

            h_t = torch.zeros(batch_size,hidden_size, dtype=torch.float).to(device)
            c_t = torch.zeros(batch_size,hidden_size, dtype=torch.float).to(device)
            h_t2 = torch.zeros(batch_size,hidden_size, dtype=torch.float).to(device)
            c_t2 = torch.zeros(batch_size,hidden_size, dtype=torch.float).to(device)

            h_t3 = torch.zeros(batch_size, hidden_size, dtype=torch.float).to(device)
            c_t3 = torch.zeros(batch_size, hidden_size, dtype=torch.float).to(device)


            for input_t in input.split(1, dim=1):


                #print(input_t.size(),'t')
               # print(input_t)

                input_t=input_t.view(-1, input_size)#############################################
               # print(input_t.size(), 't')
               # print(input_t.size(), 't')
               # print(input_t)
                input_x= self.linear1(input_t)
                ni = differential_privacy.differential_privacy_pre_2d(input_x.cpu().detach().numpy(), epsilon=100,
                                                                      client_number=3, p=1)
                # print(ni,'ni')#########################################################################
                input_x = input_x + torch.from_numpy(ni).float().to(device)

                #print(input_x.size(),'x')
               # print(input_x,'v')
                h_t, c_t = self.lstm1(input_x, (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear(torch.sigmoid(h_t3))
                #outputs += [output]
            #for i in range(future):# if we should predict the future
               # h_t, c_t = self.lstm1(output, (h_t, c_t))
                #h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                #output = self.linear(h_t2)
              #  outputs += [output]
            #outputs = torch.cat(outputs, dim=1)
            #print(output,'output')###############################################################################
            #print(output.size())################################################################
            #output=nn.functional.relu(output)
            return output


    pla=[ 7, 7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
      8,  8,  8,  8,  8,  8, 7,  7,  7,  7,  7, 7,  7,  7,  7,  7,  7,  6,
      6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  6,  6,
      9,  9,  9,  9,  9, 12, 12, 12, 12, 14, 14, 15, 15, 16, 16, 17, 17, 18,
     18, 18, 18, 18, 18, 17, 17, 16, 16, 15, 15, 15, 14, 14, 14, 14, 14, 14,
     14, 12, 12, 12, 13, 10, 10,  7,  7,  7,  4,  2,  2,  3,  2,  1,  1,  1,
      2,  3,  5,  5,  5,  8,  8, 7, 7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
      8,  8,  8,  8,  8,  8, 7,  7,  7,  7,  7, 7,  7,  7,  7,  7,  7,  6,
      6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  6,  6,
      9,  9,  9,  9,  9, 12, 12, 12, 12, 14, 14, 15, 15, 16, 16, 17, 17, 18,
     18, 18, 18, 18, 18, 17, 17, 16, 16, 15, 15, 15, 14, 14, 14, 14, 14, 14,
     14, 12, 12, 12, 13, 10, 10,  7,  7,  7,  4,  2,  2,  3,  2,  1,  1,  1,
      2,  3,  5,  5,  5,  8,  8]



    x1, y1, x2, y2=mnist_numpy.load()
    x1,y1=preprocess_rnn(x1,y1)
    x2,y2=preprocess_rnn(x2,y2)
    x2=x2[0:6000]
    y2=y2[0:6000]
    x1=x1.reshape(x1.shape[0],28,28)
    x2=x2.reshape(x2.shape[0],28,28)

    data_x_divide_train,data_y_divide_train=data_general(pla,sequence_length)
    data_x_test=data_x_divide_train[0:6000]
    data_y_test=data_y_divide_train[0:6000]

    data_x_divide_train=data_x_divide_train[0:x1.shape[0]]
    data_y_divide_train=data_y_divide_train[0:y1.shape[0]]
    z = np.concatenate((x1, data_x_divide_train), axis=2)
    zy= np.concatenate((y1, data_y_divide_train), axis=1)
    z_test=np.concatenate((x2, data_x_test), axis=2)
    zy_test=np.concatenate((y2, data_y_test), axis=1)


    x1=batch_creat(x1,x1.shape[0]/batch_size)
    y1=batch_creat(y1,y1.shape[0]/batch_size)
    x2=batch_creat(x2,x2.shape[0]/batch_size)
    y2=batch_creat(y2,y2.shape[0]/batch_size)
    data_x_divide_train=batch_creat(data_x_divide_train,data_x_divide_train.shape[0]/batch_size)
    data_y_divide_train=batch_creat(data_y_divide_train,data_y_divide_train.shape[0]/batch_size)
    data_x_test=batch_creat(data_x_test,data_x_test.shape[0]/batch_size)
    data_y_test=batch_creat(data_y_test,data_y_test.shape[0]/batch_size)

    z=batch_creat(z, z.shape[0]/batch_size)
    zy=batch_creat(zy, zy.shape[0]/batch_size)
    z_test=batch_creat(z_test, z_test.shape[0]/batch_size)
    zy_test=batch_creat(zy_test, zy_test.shape[0]/batch_size)

    model = Sequence(input_size,output_size,hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps = len(z)
    start_time = time.time()
    accs = []
    accs_tre=[]
    #print(x1.shape[0])
    iters=x1.shape[0]
    #print(iters,'iter')
    for epoch in range(num_epochs):
        for i in range(iters):
           # print(i,'i')
            # origin shape: [N, 1, 28, 28]
            a = np.zeros((model.linear1.weight.cpu().detach().numpy().shape))

            client = 0
            if client < client_number:
                z_input=z[i]
                zy_input=zy[i]
                z_input=torch.from_numpy(z_input)
                zy_input=torch.from_numpy(zy_input)
                z_input=z_input.to(device)
                zy_input=zy_input.to(device)

                #print(images[0],'image')
                #print(images,'image')
                #print(images.size())


               # print(labels,'label')
               # print(labels.size())
                # Forward pass
                outputs = model(z_input.float())
                loss = criterion(outputs,zy_input.float())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    a = model.linear1.weight.cpu().numpy() / client_number
                client = client + 1

            with torch.no_grad():


                model.linear1.weight = nn.Parameter(torch.from_numpy(a).cuda().float())

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

    #training_time = time.time() - start_time
        # Test the model
                            # In test phase, we don't need to compute gradients (for memory efficiency)
            with torch.no_grad():


                n_correct = 0
                n_samples = 0
                n_correct1=0
                n_samples1=0
                for i in range(z_test.shape[0]):

                    input_test=z_test[i]
                    label_test=zy_test[i]

                    input_test=torch.from_numpy(input_test)
                    label_test=torch.from_numpy(label_test)

                    input_test=input_test.to(device)
                    label_test=label_test.to(device)

                    label_test_image=label_test[:,0:10]#########################################
                    label_test_tre=label_test[:,10:]##############################################

                    outputs = model(input_test.float())

                    # max returns (value ,index)
                    outputs_image=outputs[:,0:10]
                    outputs_tre=outputs[:,10:]

                    _, predicted = torch.max(outputs_image.data, 1)
                    _, label_test_image_convert = torch.max(label_test_image.data, 1)
                    _, predicted_tre = torch.max(outputs_tre.data, 1)
                    _, label_test_tre_convert = torch.max(label_test_tre.data, 1)
                   # print(predicted,'pre')
                   # print(label_test,'label')

                    n_samples += label_test_image.size(0)
                    n_correct += (predicted == label_test_image_convert).sum().item()
                    n_samples1 += label_test_tre.size(0)
                    n_correct1 += (predicted_tre == label_test_tre_convert).sum().item()


                acc = 100.0 * n_correct / n_samples
                accs.append(acc)

                acc_tre=100.0 * n_correct1 / n_samples1

                accs_tre.append(acc_tre)

   
    facc = final_acc(accs_tre, acc)
    print(facc,'FedADMP')

KSP_training_time_energy()
KSP_training_acc()