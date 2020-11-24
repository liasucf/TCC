#!/usr/bin/env python
# coding: utf-8

# In[88]:
import os
import socket
import pickle
import psutil
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import time
import pandas as pd
#import syft as sy
import json 
import numpy as np
from pandas.io.json import json_normalize 
from sklearn import preprocessing
import sys
from skorch.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from numpy import array
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import mean_squared_error
import seaborn as sns
from collections import Counter
from sklearn.metrics import mean_absolute_error
from skorch import NeuralNetRegressor
from skorch.helper import predefined_split


# # Implementing LSTM Prediction of a Time Series in PyTorch

#Creating architecture of the Neural Network model
class LSTM(nn.Module):
    def __init__(self, input_size=35, n_hidden=10, n_layers=1, output_size=5):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size=n_hidden, num_layers=n_layers)
        self.hidden = self.init_hidden()
        self.linear1 = nn.Linear(n_hidden, output_size)
        self.dropout1 = nn.Dropout(p=0.5)
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.n_layers,1,self.n_hidden),
                            torch.zeros(self.n_layers,1,self.n_hidden))
    def forward(self, x): 
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(x.view(x.size(0),1, -1), self.hidden)
        lstm_out = self.dropout1(lstm_out)
        predictions = self.linear1(lstm_out.view(len(lstm_out), -1))
        return predictions


#Arguments for the models
class Arguments:
    def __init__(self):
        self.city = sys.argv[1]
        self.n_samples  = int(float(sys.argv[2]))
        self.epochs = int(float(sys.argv[3])) if len(sys.argv) > 3 else 800
        self.lr = 0.01
        self.test_batch_size = 8
        self.batch_size = 8
        self.log_interval = 10
        self.seed = 1
        self.patience = 100
        self.momentum =  0.09
        self.threshold = 0.0003
        self.layers = 1
        self.units = 10


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


args = Arguments()
torch.manual_seed(args.seed)
#Get the process that running right now
pid = os.getpid()
#use psutil to detect this process
p = psutil.Process(pid)
#Return a float representing the current system-wide CPU utilization as a percentage
#First time you call the value is zero (as a baseline), the second it will compare with the value 
#called and give a result  
p.cpu_percent(interval=None)

# ## DataSet

# ### Charging the data

with open('../Data/'+args.city+'.txt', 'r') as infile:
    data = infile.read()
    new_data = data.replace('}{', '},{')
    json_data = json.loads(f'[{new_data}]')

df = pd.DataFrame(json_data)
data = json_normalize(df['data'])
data["status"] = df["status"]


#tratement and processing the data 

data = data.drop(columns = ['attributions', 'city.url', 'debug.sync', 'time.tz','status', 'city.name', 'city.geo','time.s'])
data['aqi']=  data['aqi'].replace('-',0)
data['aqi'] = pd.to_numeric(data['aqi'])
#Labelling string values
if 'iaqi.so2.v' in data.columns:
    data = data.drop(columns = ['iaqi.so2.v'])
data["iaqi.no2.v"] = data["iaqi.no2.v"].fillna(value=data["iaqi.no2.v"].mean())

if 'forecast.daily.o3' in data.columns:
    data = data.drop(columns = ["forecast.daily.o3"])
if 'forecast.daily.pm10' in data.columns:
    data = data.drop(columns = ["forecast.daily.pm10"])
if 'forecast.daily.pm25' in data.columns:
    data = data.drop(columns = ["forecast.daily.pm25"])
if 'forecast.daily.uvi' in data.columns:
    data = data.drop(columns = ["forecast.daily.uvi"])
if 'iaqi.co.v' in data.columns:
    data = data.drop(columns = ["iaqi.co.v"])
if 'iaqi.wg.v' in data.columns:
    data = data.drop(columns = ["iaqi.wg.v"])
if 'iaqi.dew.v' in data.columns:
    data = data.drop(columns = ["iaqi.dew.v"])
    
if 'iaqi.pm25.v' not in data.columns:
    data['pm25']  = 0
if 'iaqi.pm25.v' in data.columns:
    data['iaqi.pm25.v'] = data['iaqi.pm25.v'].fillna(value=data['iaqi.pm25.v'].mean())
    data['pm25'] = data['iaqi.pm25.v']
    data = data.drop(columns = ['iaqi.pm25.v'])
#Setting the right target for prediction
data['o3'] = data['iaqi.o3.v']
data = data.drop(columns = ['iaqi.o3.v','time.v','dominentpol','idx','aqi'])
data = data.iloc[0:args.n_samples]
# ### Creating the sliding window matrix

# choose a number of time steps
n_steps_in, n_steps_out = 5, 5
# convert into input/output
X, y = split_sequences(data.values, n_steps_in, n_steps_out)



n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]
n_input = n_timesteps * n_features
X = X.reshape((X.shape[0], n_input))


# ### Train-Test Spliting and Scaling Data

#Train-Test Split
SPLIT_IDX = int(len(X) * 0.80)
X_train, X_test = X[0:SPLIT_IDX], X[SPLIT_IDX:len(X)]
y_train, y_test = y[0:SPLIT_IDX], y[SPLIT_IDX:len(X)]

#Scaling Data

scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

SPLIT_VAL = int(len(X_train) * 0.75)
size = len(X_train)
X_train, X_val = X_train[0:SPLIT_VAL], X_train[SPLIT_VAL:size]
y_train, y_val = y_train[0:SPLIT_VAL], y_train[SPLIT_VAL:size]


# # Train

# ### Creating tensors for the dataset to be used in Pytorch


X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).float()

#Creating the tensor datasets
Val =  TensorDataset(X_val, y_val)

#Chosing the host (server) ip and port to connect to

host = "172.18.0.2"
port = 8000

#Making a socket for communication and connecting with the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
s.connect((host,port))
number = 0

#A forever loop to always be listening to the server 
while True:
    msg = 0
    #Receiving the model from the server
    start_time = time.time()
    print('Connected to server', '\n')
    #Message received
    msg = int.from_bytes(s.recv(4), 'big')

    #Saving the message received with the models parameters 
    print("Saving the model received", '\n')
    f = open('model_rec.sav','wb')
    while msg:
        # until there are bytes left...
        # fetch remaining bytes or 4094 (whatever smaller)
        rbuf = s.recv(min(msg, 4096))
        msg -= len(rbuf)
        # write to file
        f.write(rbuf)
    f.close()
    print('Time to receive the model from the server',time.time() - start_time , '\n')
    
    #Loading the model that was saved and received from server
    model = pickle.load(open('model_rec.sav', 'rb'))
    
    #Defining the ealy stopping method
    early = EarlyStopping(patience=args.patience, threshold= args.threshold )

    #Using the model with the NeuralNetRegressor to configure parameters
    net = NeuralNetRegressor(
    model,
    max_epochs=args.epochs,
    lr=args.lr,
    batch_size = args.batch_size,
    optimizer__momentum=args.momentum,
    train_split=predefined_split(Val),
    iterator_train__shuffle=False,
    iterator_valid__shuffle=False,
    callbacks=[early])
    
    start_training = time.time()

    #Training 
    net.fit(X_train, y_train)
    #saving the training time
    b = open("train_temps.txt", "a+")
    b.write("Iteration: " + str(number) + " Time to train: " + str(time.time() - start_training) + 'in '+ str(args.epochs)  + 'epoches'  + '\n' )
    b.close()
    
    
    # visualize the loss as the network trained
    # plotting training and validation loss
    epochs = [i for i in range(len(net.history))]
    train_loss = net.history[:,'train_loss']
    valid_loss = net.history[:,'valid_loss']
    
    fig = plt.figure(figsize=(25,10))
    plt.plot(epochs,train_loss,'g-');
    plt.plot(epochs,valid_loss,'r-');
    plt.title('Training Loss Curves');
    plt.xlabel('Epochs');
    plt.ylabel('Mean Squared Error');
    plt.legend(['Train','Validation']);
    fig.savefig('loss_plot'+str(number)+'.png', bbox_inches='tight')
    
    
    #Testing model
    y_pred = net.predict(X_test)
    a = open("test_losses.txt", "a+")
    a.write("Number: " + str(number) + '\n')
    a.write("MSE loss: " + str(mean_squared_error(y_test, y_pred)) + " MAE loss: " + str(mean_absolute_error(y_test, y_pred))  + '\n' )
    a.close()
    
    target = scaler.inverse_transform(y_pred)
    real = scaler.inverse_transform(y_test)

    fig1 = plt.figure(figsize=(16,7))
    plt.plot(real[:,0], color = 'black', label = 'Target')
    plt.plot(target[:,0], color = 'green', label = 'Predicted')
    plt.xlabel('Time (H)')
    plt.ylabel('AQI O3')
    plt.title('O3 variation over time')
    plt.legend()
    fig1.savefig('prediction_plot'+str(number)+'.png', bbox_inches='tight')
    
    ## Classification due to trend
    
    df_evaluation = pd.DataFrame(real[:,0], columns =['Real']) 
    df_evaluation['Target'] = target[:,0]
    
    
    #1 if increasing
    #0 stopped 
    #-1 if decreasing
    result = map(lambda x, y : 1 if x > y else (-1 if x < y else 0),  df_evaluation['Real'].shift(-1),  df_evaluation['Real']) 
    df_evaluation['Real_Trend'] = list(result)
    
    
    result_target = map(lambda x, y : 1 if x > y else (-1 if x < y else 0),  df_evaluation['Target'].shift(-1),  df_evaluation['Target']) 
    df_evaluation['Target_Trend'] = list(result_target)
    # ### Precision Recall
    
    precision, recall, fscore, support = score(df_evaluation['Real_Trend'], df_evaluation['Target_Trend'])
    accuracy = accuracy_score(df_evaluation['Real_Trend'], df_evaluation['Target_Trend'])
    d = open("classification.txt", "a+")
    d.write("Number: " + str(number) + '\n')
    d.write('accuracy: {}'.format(accuracy))
    d.write('precision: {}'.format(precision))
    d.write('recall: {}'.format(recall))
    d.write('fscore: {}'.format(fscore))
    d.write('support: {}'.format(support))
    
    d.write('Counter Real Trend: {}'.format(Counter(df_evaluation['Real_Trend'])))
    d.write('Counter Predicted Trend: {}'.format(Counter(df_evaluation['Target_Trend'])))
    d.close()
    # ### Confusion Matrix 
    
    cm = confusion_matrix(df_evaluation['Real_Trend'], df_evaluation['Target_Trend'])
    df_cm = pd.DataFrame(cm, columns=np.unique(df_evaluation['Real_Trend']), index = np.unique(df_evaluation['Real_Trend']))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.set(font_scale=1.4)#for label size
    svm = sns.heatmap(df_cm, cmap="Blues", annot=True)
    figure = svm.get_figure()    
    figure.savefig('svm_conf.png', dpi=400)
     
    #Collecting the information of memory and CPU usage
    z = open("memory_cpu.txt", "a+")
    z.write("Number: " + str(number) + '\n')
    z.write('percentage of memory use: '+ str(p.memory_percent())+ '\n')
    z.write('percentage utilization of this process in the system '+ str(p.cpu_percent(interval=None))+ '\n')
    z.close()
    
    #Saving the model updated
    number = number + 1
    filename = 'model.sav'
    pickle.dump(model, open(filename, 'wb'))
    print(model)
    print(model.parameters())

    #Sending the updated model to the server    
    with open("model.sav", "rb") as r:
                
        print("sending the updated model",  '\n')
        data = r.read()
        # check data length in bytes and send it to client
        data_length = len(data)
        s.send(data_length.to_bytes(4, 'big'))
        s.send(data)
        print('Time to send updated model to the server',time.time() - start_time , '\n')
    r.close()
    
    
        
# close the connection  
s.close() 








