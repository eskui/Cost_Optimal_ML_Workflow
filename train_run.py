import numpy as np
import pandas as pd
import os
import tensorflow as tf
from datetime import datetime
from datetime import timedelta
#import re
#import matplotlib.pyplot as plt
import support.ts_class as ts_class
import support.load_and_process_data as lpdata
import math
import pyarrow.parquet as pq
import sys

def compile_and_fit(model, window, max_epochs):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience = 10, restore_best_weights = True)

    model.compile(loss=tf.losses.MeanSquaredError(),optimizer=tf.optimizers.Adam(),metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs,validation_data=window.val, callbacks=[early_stopping])
    return history

def main():
    print("training sciprt starts....")
    #sys_argv[1] = first year for trianing data
    train_years = list(range(int(sys.argv[1]),2021))
    val_year = [2021]
    test_year = [2022]

    MAX_EPOCHS = int(sys.argv[2])
    WIND_SIZE = int(sys.argv[3])
    LABELS_TO_PREDITC = ['MSFT_Close']

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

    print(train_set.shape)
    print(val_set.shape)
    print(test_set.shape)

    day_window = ts_class.WindowGenerator(input_width=WIND_SIZE, label_width=1, shift=1,label_columns=LABELS_TO_PREDITC,train_df=train_set, val_df=val_set, test_df=test_set)

    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(WIND_SIZE,),
                               activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])

    history = compile_and_fit(conv_model, day_window, MAX_EPOCHS)

if __name__ == "__main__":
    main()
