#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 18:03:54 2019

@author: Meng
"""
#data downloaded from hackerRank
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
import numpy as np


with open('input01.txt') as f:
    X = f.readlines()
#you may also want to remove whitespace characters like `\n` at the end of each line
X = [int(element.strip()) for element in X] 
X = np.array(X)   
size = int(len(X)*0.66)
train, test= X[0:size], X[size:len(X)]
history= list(train)
predictions= list()

for t in range(len(test)):
    model = ARIMA(history, order=(6,1,1))
    model_fit=model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    obs=test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
    

with open('output01.txt') as f:
    y = f.readlines()
#you may also want to remove whitespace characters like `\n` at the end of each line
y = [int(element.strip()) for element in y] 
y = np.array(y)
Y= y
#pyplot.plot(model_fit.forecast(steps=30)[0],'r')
#pyplot.plot(predictions,'r')
#pyplot.plot(model_fit.forecast(steps=30)[0], 'b'); pyplot.plot(Y,'r')
pyplot.plot(model_fit.forecast(steps=30)[0], 'b'); pyplot.plot(Y,'r'); pyplot.plot(model_fit.predict(len(X), len(X)+30),'k')
###
my_list=list(map(int,input().rstrip().split()))
#or:
numbers = input("Enter numbers: ").split()
numbers = [int(x) for x in numbers]