import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from yahoo_fin import stock_info as si
from tqdm import tqdm
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import random
import math

import requests
import lxml.html as html
import time
import string
import json

evaluate_bool = True


# set seed, so we can get the same results after rerunning several times
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)


def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1,
              test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(
                np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
    # this last_sequence will be used to predict in future dates that are not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    # shift the last sequence by -1
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    # split the dataset
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle)
    # return the result
    return result


def create_model(sequence_length, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                 loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(
                    cell(units, return_sequences=True), input_shape=(None, sequence_length)))
            else:
                model.add(cell(units, return_sequences=True,
                               input_shape=(None, sequence_length)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=[
                  "mean_absolute_error"], optimizer=optimizer)
    return model


# Window size or the sequence length
N_STEPS = 100
# Lookup step, 1 is the next day
LOOKUP_STEP = 1
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")
# model parameters
N_LAYERS = 3
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False
# training parameters
# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 500
# Apple stock market

# Model name trained on AAPL
model_name = f"AAPL"
if BIDIRECTIONAL:
    model_name += "-b"
# construct the model
model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                     dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

# TODO: LOOP HERE

total_listings = requests.post("https://www.nyse.com/api/quotes/filter", data=json.dumps({
    "instrumentType": "EQUITY", "pageNumber": 0, "sortColumn": "NORMALIZED_TICKER",
    "sortOrder": "ASC", "maxResultsPerPage": 1, "filterToken": ""}), headers={
    'Content-Type': 'application/json'
}).json()[0]['total']

time.sleep(1)

stocks_json = requests.post("https://www.nyse.com/api/quotes/filter", data=json.dumps({
    "instrumentType": "EQUITY", "pageNumber": 0, "sortColumn": "NORMALIZED_TICKER",
    "sortOrder": "ASC", "maxResultsPerPage": total_listings, "filterToken": ""}), headers={
    'Content-Type': 'application/json'
}).json()

data_arr = []
for i in tqdm(range(len(stocks_json))):
    try:
        #ticker = "XLE"
        ticker = stocks_json[i]['symbolExchangeTicker']
        ticker_data_filename = os.path.join(
            "data_all", f"{ticker}_{date_now}.csv")
        # model name to save, making it as unique as possible based on parameters
        # model_name = f"{date_now}_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"

        try:
            data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                             feature_columns=FEATURE_COLUMNS, shuffle=False)
        except:
            continue

        # evaluate the model
        mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
        # calculate the mean absolute error (inverse scaling)
        mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[
            0][0]
        # print("Mean Absolute Error:", mean_absolute_error)

        def predict(model, data, classification=False):
            # retrieve the last sequence from data
            last_sequence = data["last_sequence"][:N_STEPS]
            # retrieve the column scalers
            column_scaler = data["column_scaler"]
            # reshape the last sequence
            last_sequence = last_sequence.reshape(
                (last_sequence.shape[1], last_sequence.shape[0]))
            # expand dimension
            last_sequence = np.expand_dims(last_sequence, axis=0)
            # get the prediction (scaled from 0 to 1)
            prediction = model.predict(last_sequence)
            # get the price (by inverting the scaling)
            predicted_price = column_scaler["adjclose"].inverse_transform(prediction)[
                0][0]
            return predicted_price

        # predict the future price
        try:
            future_price = predict(model, data)
        except:
            continue
        # print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")

        stocks_held = []
        cash = 10000
        days_to_trade = 365

        y_test = data['y_test']
        X_test = data['X_test']
        y_pred = model.predict(X_test)

        y_test = y_test[-1*days_to_trade:]
        X_test = X_test[-1*days_to_trade:]
        y_pred = y_pred[-1*days_to_trade:]

        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(
            np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]
                            ["adjclose"].inverse_transform(y_pred))
        #
        # Model Buying / Selling
        #

        # Bad Actions
        for x in range(len(y_test)):
            current_price = y_test[x]
            tomorrows_price = y_pred[x]

            if tomorrows_price < current_price:
                # Buy
                if cash > current_price:
                    try:
                        stocks_amt = math.floor(cash/current_price)
                        cash = cash % current_price
                        for s in range(stocks_amt):
                            stocks_held.append({'price': current_price})
                    except:
                        continue

                if len(stocks_held) != 0 and tomorrows_price > current_price or len(stocks_held) != 0 and x == len(y_test)-1:
                    # Sell
                    offset = 0

                    for x in range(len(stocks_held)):
                        cash += current_price
                        stocks_held.pop(x+offset)
                        offset -= 1

        # print("Bad Model CASH LEFT: {} on {}".format(str(cash), ticker))
        bad_cash = cash

        cash = 10000
        stocks_held = []
        for x in range(len(y_test)):
            current_price = y_test[x]
            tomorrows_price = y_pred[x]

            if tomorrows_price > current_price:
                # Buy
                if cash > current_price:
                    try:
                        stocks_amt = math.floor(cash/current_price)
                        cash = cash % current_price
                        for s in range(stocks_amt):
                            stocks_held.append({'price': current_price})
                    except:
                        continue

            if len(stocks_held) != 0 and tomorrows_price < current_price or len(stocks_held) != 0 and x == len(y_test)-1:
                # Sell
                offset = 0

                for x in range(len(stocks_held)):
                    cash += current_price
                    stocks_held.pop(x+offset)
                    offset -= 1

        # print("Good Model CASH LEFT: {} on {}".format(str(cash), ticker))

        def plot_graph(model, data):
            y_test = data["y_test"]
            X_test = data["X_test"]
            y_pred = model.predict(X_test)
            y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(
                np.expand_dims(y_test, axis=0)))
            y_pred = np.squeeze(data["column_scaler"]
                                ["adjclose"].inverse_transform(y_pred))
            # last 200 days, feel free to edit that
            plt.plot(y_test[-365:], c='b')
            plt.plot(y_pred[-365:], c='r')
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.legend(["Actual Price", "Predicted Price"])
            plt.show()

        # plot_graph(model, data)

        def get_accuracy(model, data):
            y_test = data["y_test"]
            X_test = data["X_test"]
            y_pred = model.predict(X_test)
            y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(
                np.expand_dims(y_test, axis=0)))
            y_pred = np.squeeze(data["column_scaler"]
                                ["adjclose"].inverse_transform(y_pred))
            y_pred = list(map(lambda current, future: int(float(future) > float(
                current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
            y_test = list(map(lambda current, future: int(float(future) > float(
                current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
            return accuracy_score(y_test, y_pred)

        # print(str(LOOKUP_STEP) + ":", "Accuracy Score:", get_accuracy(model, data))
        data_arr.append({
            'ticker': ticker,
            'bad_cash': bad_cash,
            'good_cash': cash
        })
    except:
        continue

with open("results.json", 'w+') as o:
    json.dump({
        'data': data_arr
    }, o)
