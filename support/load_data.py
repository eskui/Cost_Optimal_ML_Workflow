import numpy as np
import pandas as pd
import os
from datetime import datetime
import math
import csv
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
