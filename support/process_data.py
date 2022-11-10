import numpy as np
import pandas as pd
import os
import tensorflow as tf
from datetime import datetime
import math
import csv
from time import sleep
from datetime import timedelta


def clean_feature(string):
    result = re.sub(r'\(\w+\)$','',string.strip())
    return result.strip()

def validate_date(d):
    try:
        datetime.strptime(d, '%A, %B %d, %Y')
        return True
    except ValueError:
        return False

def validate_time(d):
    try:
        datetime.strptime(d, '%H:%M')
        return True
    except ValueError:
        return False

def validate_datetime(d):
    try:
        datetime.strptime(d, '%d/%m/%Y %H:%M:%S')
        return True
    except ValueError:
        return False

def cast_value(value):
    try:
        return float(value)
    except:
        value = value.replace(',','')
        if value.endswith('k') or value.endswith('K'):
            return 10**3*float(value[0:(len(value)-1)])
        if value.endswith('m') or value.endswith('M'):
            return 10**6*float(value[0:(len(value)-1)])
        if value.endswith('b') or value.endswith('B'):
            return 10**9*float(value[0:(len(value)-1)])
        if value.endswith('%'):
            return 10**(-2)*float(value[0:(len(value)-1)])
        if value.endswith('T'):
            return 10**(12)*float(value[0:(len(value)-1)])
        else:
            return None

def detrend(data):
    x = data.to_numpy().reshape(-1,1)
    y = np.arange(0,len(data))
    model = LinearRegression()
    model.fit(x, y)
    trend = model.predict(x)
    detrended = y-trend
    return detrended

def trending(y, x, model):
    trend = model.predict(x)
    return (y + trend)

def clean(data, outlier_remove = True):
    print("Remove outliers...")

    K = 6

    data = data.dropna(axis = 1,how = "all")

    for i,column in enumerate(data):
        values = data.loc[data[column].notnull(),column]

        if outlier_remove:
            data.loc[data[column].notnull(),column] = IQR_outlier_remove(values,K).fillna(method="bfill")

    return data
