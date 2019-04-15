"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import mnist, cifar10
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation ,Input , TimeDistributed, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from datetime import datetime
import pandas as pd
import numpy as np
import plotly as py
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import talib
import math

"""class Train():
    #Class that implements genetic algorithm for LSTM optimization
    def __init__(self, path=None):
        Create an optimizer.
        Args:
            nn_param_choices (dict): Possible network paremters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated

        self.path = path
"""
    #global path, raw, openprice, highprice, lowprice, closeprice, avg_price, data, avg_price, sma_240, bollupper,bollmiddle, bolllower, ATR

    # Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def get_cifar10():
        # Retrieve the CIFAR dataset and process the data.
        # Set defaults.
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

        # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_mnist():
    #Retrieve the MNIST dataset and process the data.
    # Set defaults.
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def getdata(path):

    """Read data from CSV"""

    raw = pd.read_csv(path,usecols=[0,1,2,3,4,5],skiprows=1)
    #path = file location from main()
    raw.columns = ['Date','Open','High','Low','Close','Volumn']
    raw['Date'] = raw['Date'].apply(lambda y: datetime.strptime(y,'%d.%m.%Y %H:%M:%S.000'))
    raw = raw[(raw['Date'].dt.year >=2015)] #data range(2011-2018)
    raw = raw.drop_duplicates(keep=False)
    print('INFO:All data is extracted successfully')

    return raw

def candlebar_analysis(raw):

    openprice=raw['Open'].astype(float).values
    highprice=raw['High'].astype(float).values
    lowprice=raw['Low'].astype(float).values
    closeprice=raw['Close'].astype(float).values
    avg_price = talib.AVGPRICE(opprice, hiprice, lowprice, closeprice)
    print('INFO:Candle Parameters is created')


    return openprice, highprice, lowprice, closeprice, avg_price

def technical_analysis(openprice, highprice, lowprice, closeprice, avg_price):

    sma_240 = talib.SMA(opprice, timeperiod=240) #sma_240 = 10 Days Avg
    ATR =talib.ATR(hiprice,lowprice,closeprice,timeperiod=24)
    #RSI = talib.RSI(closeprice,120)
    bollupper,bollmiddle,bolllower = talib.BBANDS(closeprice,nbdevup=2, nbdevdn=2, matype=MA_Type.T3)
    print ('INFO: Technical Parameters is calculated')

    return avg_price, sma_240, bollupper, bollmiddle, bolllower, ATR

def series_to_supervised(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def reshaping(avg_price, sma_240, bollupper,bollmiddle, bolllower, ATR):

    batch_size = 1000
    nb_classes = 48

    dataset = pd.DataFrame(data=[avg_price, sma_240, bollupper,bollmiddle, bolllower, ATR]).transpose()
    dataset.columns = ['avg_price', 'SMA240', 'BBupper', 'BBMiddle' ,'BBLower', 'ATR']  # NAMING COLUMN
    dataset.dropna(inplace=True)

    data = dataset.values
    data = data.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    data=series_to_supervised(data,1,1)
    data.drop(data.columns[[7,8, 9, 10, 11]], axis=1, inplace=True)
    data = data.values #DF to list
    train_size = int(len(data)*0.7) #70% train ,30% test
    trainset = data[: train_size, :]
    testset = data[train_size :, :]
    x_train, y_train = trainset[:, :-1], trainset[:, -1]
    x_test, y_test = testset[:, :-1], testset[:, -1]
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    input_shape = (x_train.shape[1], x_train.shape[2])

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    # indicate wt inside Sequential
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']
    dropout = network['dropout']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(LSTM(nb_neurons, activation=activation, return_sequences=True))
            model.add(LSTM(nb_neurons, return_sequences=True))

        model.add(Dropout(dropout))

    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='mean_squared_error', optimizer=optimizer,
                  metrics=['mse', 'mae', 'mape'])

    return model

def train_and_score(network, dataset, path):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'AUDUSD60':
        raw = getdata(path)

        topenprice, highprice, lowprice, closeprice, avg_price = candlebar_analysis(raw)

        avg_price, sma_240, bollupper, bollmiddle, bolllower, ATR = technical_analysis(openprice, highprice, lowprice, closeprice, avg_price)

        nb_classes, batch_size, input_shape, x_train, \
                x_test, y_train, y_test = reshaping(avg_price, sma_240, bollupper,bollmiddle, bolllower, ATR)

    model = compile_model(network, nb_classes, input_shape)

    history = model.fit(x_train, y_train,
                        epochs=epoch,
                        batch_size=batch_size,
                        verbose=2,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stopper])

    trainPredict = model.predict(x_train)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[2]))  # reshape the data
    trainresult = np.concatenate((trainPredict, x_train[:, 1:]), axis=1)

    trainScore = math.sqrt(mean_squared_error(y_train, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))

    #calculate the SD SCORE
    trainresult = scaler.inverse_transform(trainresult)
    trainPredict = trainresult[:, 0]

    testPredict = model.predict(x_test)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))  #reshape the data
    testresult = np.concatenate((testPredict, x_test[:, 1:]), axis=1)

    testScore = math.sqrt(mean_squared_error(y_test, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))

    testresult  = scaler.inverse_transform(testresult)
    testPredict = testresult[:,0]

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.
