import numpy as np
import pandas as pd
import os
import tensorflow as tf
from datetime import datetime
from datetime import timedelta

import support.ts_class as ts_class
import math
import pyarrow.parquet as pq
import sys
from timeit import default_timer as timer
from datetime import timedelta

import support.paleo_estimator as paleo

import time
import datetime

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

LABELS_TO_PREDITC = ['MSFT_Close']


def main():
    val_year = [2021]
    test_year = [2022]
    train_years = list(range(int(sys.argv[1]),2021))
    WIND_SIZE = int(sys.argv[2])
    device_name = sys.argv[3]
    NO_OF_BATCHES = int(sys.argv[4])

    train_set = pq.read_table('./train_data', filters=[('year','in',train_years)])\
        .to_pandas()\
        .drop(columns = ["year"])\
        .set_index("Datetime")

    val_set = pq.read_table('./train_data', filters=[('year','in',val_year)])\
        .to_pandas()\
        .drop(columns = ["year"])\
        .set_index("Datetime")

    test_set = pq.read_table('./train_data', filters=[('year','in',test_year)])\
        .to_pandas()\
        .drop(columns = ["year"])\
        .set_index("Datetime")

    window = ts_class.WindowGenerator(input_width=WIND_SIZE, label_width=1, shift=1,label_columns=LABELS_TO_PREDITC,train_df=train_set, val_df=val_set, test_df=test_set)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(WIND_SIZE,),
                               activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False
        )

    model.compile(loss=tf.losses.MeanSquaredError(),optimizer=optimizer,metrics=[tf.metrics.MeanAbsoluteError()])

    time_callback = TimeHistory()
    model.fit(window.train, epochs=1,validation_data=window.val, steps_per_epoch=NO_OF_BATCHES, callbacks=[time_callback])
    times = time_callback.times

    single_batch_time = times[0]*1000/NO_OF_BATCHES
    single_batch_time_paleo = paleo.estimate_required_time(device_name)
    ct = datetime.datetime.now()
    print("Average execution time of one batch was (ms):\n{0} \nPALEO estimation:\n{1}".format(single_batch_time,single_batch_time_paleo))

    f = open("train_model_results.csv", "a")
    f.write("{0},{1},{2},{3},{4},{5}\n".format(ct,single_batch_time,single_batch_time_paleo,device_name,NO_OF_BATCHES,WIND_SIZE))
    f.close()

if __name__ == "__main__":
    main()
