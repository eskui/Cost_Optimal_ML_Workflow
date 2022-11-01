import numpy as np
import pandas as pd
import os
import tensorflow as tf
from datetime import datetime
import re
#from oauth2client.service_account import ServiceAccountCredentials
#import gspread
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from scipy import signal
from sklearn.linear_model import LinearRegression
import investpy
import math
import csv
import requests
import progressbar
from time import sleep
from datetime import timedelta
import gspread

def open_google_sheet(file):
    gc = gspread.service_account()

    wks = gc.open(file).sheet1

    data = wks.get_all_records()
    #headers = data.pop(0)

    #df = pd.DataFrame(data, columns=headers)
    return data


def clean_feature(string):
    result = re.sub(r'\(\w+\)$','',string.strip())
    return result.strip()

def IQR_outlier_remove(data,k):
    q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q75 - q25
    cut_off = iqr * k
    lower, upper = q25 - cut_off, q75 + cut_off
    data.iloc[(data < lower) | (data > upper)] = None
    return data

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

def standardize(x, mu = np.nan, std = np.nan):
    if mu == np.nan:
        std_value = (x-np.mean(x))/np.std(x)
    else:
        std_value = (x-mu)/std
    return std_value

def inv_standardize(z,mu,std):
    return z*std + mu

def detrend(data):
    x = data.iloc[:,0].to_numpy().reshape(-1,1)
    y = data.iloc[:,1].to_numpy()
    model = LinearRegression()
    model.fit(x, y)
    trend = model.predict(x)
    detrended = y-trend
    return detrended, model

def trending(y, x, model):
    trend = model.predict(x)
    return (y + trend)

def calculate_hitrate(real,pred):
    correct = 0
    counted = 0
    prev_index = None
    for index,value in pred.items():
        if prev_index == None:
            prev_index = index
            continue
        pred_change = pred[index] > pred[prev_index]
        real_change = real[index] > real[prev_index]
        #print(pred_change,real_change)
        if (pred_change == real_change):
            correct += 1
        prev_index = index
    return correct/len(pred)

def add_econom_data(countr, start_date, end_date, imp, features = None):
    print("Retrieving economic data...")

    feature_type = "actual"

    start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    start_date = datetime.strftime(start_date, '%d/%m/%Y')
    #print(start_date)

    end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strftime(end_date, '%d/%m/%Y')
    #print(end_date)

    raw = investpy.news.economic_calendar(countries=countr, from_date = start_date, to_date = end_date, importances = imp, time_zone = 'GMT -4:00')

    if features == None:
        features = [None]*len(raw["event"])
        for i,feat in enumerate(raw["event"]):
            features[i] = clean_feature(feat)
        features = set(features)

    #features.add("timestamp")
    results = pd.DataFrame(data=np.nan,columns=features,dtype='float', index=pd.date_range(start_date,end_date, freq="min"))

    prev_tt_dic = {}
    prev_actual_dic = {}
    count = 0
    for index,row in raw.iterrows():

        feature = clean_feature(row["event"])
        if feature not in features:
            continue

        dtstamp = row["date"]+" "+row["time"]

        try:
            timestamp = datetime.strptime(dtstamp+":00", '%d/%m/%Y %H:%M:%S')
        except ValueError:
            continue

        if timestamp >= datetime.today():
            feature_type = "forecast"
        else:
            feature_type = "actual"

        if row[feature_type] == None or "YoY" in row["event"]:
            continue
        else:
            cast = cast_value(row[feature_type])
            actual = cast[0]

            if not(cast[1]):
                print(timestamp, row["event"])
                continue



            try:
                prev_index = prev_tt_dic[feature]
                prev_value = prev_actual_dic[feature]

            except KeyError:
                prev_tt_dic[feature] = timestamp
                prev_actual_dic[feature] = actual

            else:
                time_delta = timestamp - prev_tt_dic[feature]
                minutes_delta = time_delta.days*24*60+time_delta.seconds/60
                results.loc[prev_tt_dic[feature]:timestamp,feature] = np.linspace(prev_actual_dic[feature],actual,len(results.loc[prev_tt_dic[feature]:timestamp,feature]))
                prev_tt_dic[feature] = timestamp
                prev_actual_dic[feature] = actual

    #results.dropna(axis=1, how='all', inplace = True)

    return results, prev_tt_dic

def clean(data, outlier_remove = True):
    print("Remove outliers...")

    K = 6

    data = data.dropna(axis = 1,how = "all")

    for i,column in enumerate(data):
        values = data.loc[data[column].notnull(),column]

        if outlier_remove:
            data.loc[data[column].notnull(),column] = IQR_outlier_remove(values,K).fillna(method="bfill")

        #data[column] = scaler.fit_transform(data[column].to_numpy().reshape(-1, 1))

    return data

def clean_and_scale_std(data, outlier_remove = True):
    print("Clean, augment, and transform economic data...")

    K = 6
    TREND_TRESHOLD = 90

    data = data.dropna(1,"all")
    mus = {}
    stds = {}
    trend_models = {}

    count = 0
    with progressbar.ProgressBar(data.shape[1]) as bar:
        for column in data:
            values = data.loc[data[column].notnull(),column]
            detrended, model, trend = detrend(values)
            if values[0] < values[-1] and np.percentile(values,TREND_TRESHOLD) < trend[-1] or values[0] > values[-1] and np.percentile(values,TREND_TRESHOLD) > trend[-1]:
                data.loc[data[column].notnull(),column] = detrended
                trend_models[column] = model

            elif outlier_remove:
                data.loc[data[column].notnull(),column] = IQR_outlier_remove(values,K).fillna(method="bfill")

            mu = np.mean(data[column])
            std = np.std(data[column])
            data[column] = standardize(data[column],mu,std)
            mus[column] = mu
            stds[column] = std
            #tallenna mu ja std dictiin
            bar.update(count)
            count += 1
    data.fillna(value=0, inplace = True)

    return data, stds, mus, trend_models

def generate_volumebars_to_file(data_path,period, frequency, cs = 1000000):
    df_chunk = pd.read_csv(data_path, chunksize = cs)
    not_first = False
    batch = 1
    for results in df_chunk:
        results.index = pd.to_datetime(results["Unnamed: 0"], format = '%Y-%m-%d %H:%M:%S')
        results.drop(["Unnamed: 0"], axis=1, inplace = True)
        results = results.assign(timedelta = 0)
        if not_first:
            results = pd.concat([left_over,results])
        idx_to_drop = set()
        prev_index = results.index[0]
        cum_vol = 0
        with progressbar.ProgressBar(max_value=len(results.index)) as bar:
            for count,index in enumerate(results.index):
                volume = results.loc[index,"Volume"]
                cum_vol += volume
                if index == results.index[-1]:
                    left_over = results
                    not_first = True
                elif cum_vol < frequency:
                    idx_to_drop.add(index)
                elif cum_vol >= frequency:
                    results = results.drop(idx_to_drop)
                    time_delta = index - prev_index
                    minutes_delta = time_delta.days*24*60+time_delta.seconds/60
                    results.loc[index,"timedelta"] = minutes_delta
                    idx_to_drop = set()
                    cum_vol = 0
                    prev_index = index
                bar.update(count)
        path_name = "data/vb/"+str(frequency)+"/"+period+"/"
        file_name = str(batch)+"_vb_sampled.csv.zip"
        print("Write to file "+path_name+file_name+", last index", index)
        os.makedirs(path_name, exist_ok=True)
        results.to_csv(path_name+file_name)
        batch += 1

def determine_volumebars(volumes, frequency = 5000000):

    prev_index = volumes.index[0] - timedelta(minutes=1)
    minutes_delta = pd.Series(data=np.nan, index=pd.date_range(volumes.index[0],volumes.index[-1], freq="min"))
    cum_vol = 0
    with progressbar.ProgressBar(max_value=len(volumes.index)) as bar:
        for count,index in enumerate(volumes.index):
            volume = volumes[index]
            cum_vol += volume
            if cum_vol < frequency:
                continue
            elif cum_vol >= frequency:
                time_delta = index - prev_index
                minutes_delta[index] = time_delta.days*24*60+time_delta.seconds/60
                cum_vol = 0
                prev_index = index
            bar.update(count)
            minutes_delta.dropna(inplace = True)
    return minutes_delta

def read_vbs(folder):
    first = True
    d = {}
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.name[0] == ".":
                continue
            else:
                d[int(re.search(r'\d*',entry.name).group(0))] = entry.name

    files = pd.Series(d)
    files.sort_index(inplace = True)

    for index, value in files.items():
        if first:
            vb = pd.read_csv(folder+value)
            first = False
        else:
            vb = pd.concat([vb,pd.read_csv(folder+value)])
    vb.drop(['Volume'], axis=1, inplace = True)
    vb.index = pd.to_datetime(vb["Unnamed: 0"], format = '%Y-%m-%d %H:%M:%S')
    vb.drop(["Unnamed: 0"], axis=1, inplace = True)
    vb.index.rename("Datetime", inplace = True)
    return vb

def compile_and_fit(model, window, patience=2, max_epochs=30):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience,mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),optimizer=tf.optimizers.Adam(),metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs,validation_data=window.val)
    return history

def make_predictions(model, data,window):
    s = len(data)
    i = 0
    test_batch = np.array(data.iloc[i:i+window,:])[np.newaxis,:]
    i += 1

    while i < s:
        next_batch = np.array(data.iloc[i:i+window,:])[np.newaxis,:]
        if test_batch.shape[1] == next_batch.shape[1]:
            test_batch = np.concatenate((test_batch,next_batch))
        else:
            break
        i += 1

    return model(test_batch)

def av_get_intraday(sb,iv,years,months):

    print("Retrieve 1min intraday data from Alphavantage...")
    API_KEY_AV = pd.read_csv('keys.csv').loc[0,"API_KEY_AV"]

    years = np.arange(years)+1
    months = np.arange(months)+1
    df = pd.DataFrame()
    calls = 0

    for y in years:
        with progressbar.ProgressBar(max_value=len(months)) as bar:
            for m in months:
                if calls > 4:
                    sleep(60)
                    calls = 0
                sl = 'year'+str(int(y))+'month'+str(int(m))
                # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
                CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol='+sb+'&interval='+iv+'&slice='+sl+'&apikey='+API_KEY_AV

                with requests.Session() as s:
                    download = s.get(CSV_URL)
                    decoded_content = download.content.decode('utf-8')
                    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
                    #cr = pd.read_csv(decoded_content.splitlines(), delimiter=',')
                    l = (list(cr))
                    df = pd.concat((df,pd.DataFrame(l[1:])))
                bar.update(m)
                calls += 1

    df.rename(columns={0:'Datetime', 1:'Open', 2:'High', 3:'Low', 4:'Close', 5:'Volume'},inplace = True)

    df.index = pd.to_datetime(df["Datetime"], format = '%Y-%m-%d %H:%M:%S')
    df.drop(["Datetime"], axis=1, inplace = True)

    df["Ticker"] = sb

    return df.sort_index().astype({'Open':'float64', 'High':'float64', 'Low':'float64', 'Close':'float64', 'Volume':'int64'})

def retrieve_data(files, verbose = False):
    print("Retrieving economic data...")
    results = {}
    new_idx = 0

    for f in files:
        raw = open_google_sheet(f)
        raw = pd.DataFrame(raw)

        for index,row in raw.iterrows():
            if validate_date(row["Time"]):
                day = datetime.strptime(row["Time"], '%A, %B %d, %Y')
            elif not(validate_time(row["Time"])):
                if verbose:
                    print("skipping row:",day, row["Event"])
                continue
            else:
                if row["Actual"] == '' or "YoY" in row["Event"]:
                    continue
                else:
                    time = datetime.strptime(row["Time"], '%H:%M').time()
                    timestamp = datetime.combine(day,time)
                    event = clean_feature(row["Event"])
                    actual = cast_value(row["Actual"])
                    forecast = cast_value(row["Forecast"])
                    previous = cast_value(row["Previous"])
                    results[new_idx] =[timestamp,event,actual,forecast,previous]
                    new_idx += 1
    return pd.DataFrame.from_dict(results, orient='index', columns=["Datetime", "Event","Actual","Forecast","Previous"])
