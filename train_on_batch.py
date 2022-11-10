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
from timeit import default_timer as timer
from datetime import timedelta
import paleo.profiler as profilers
import paleo.device as device

NETWORK = device.Network("EC2_network", 20)
DEVICE_T4GPU = device.Device("NVIDIA_T4",585,8100,320,True)
DEVICE_LOCAL = device.Device("local_CPU",2300,40,2133,False)

def estimate_required_time(device_name):
    if device_name == "NVIDIA_T4":
        device = DEVICE_T4GPU
    elif device_name == "local_CPU":
        device = DEVICE_LOCAL
    else:
        print("Please give a proper device and try again")
        exit()

    profiler = profilers.BaseProfiler("conv_network.json", device, NETWORK)
    forward_time, kbytes = profiler.estimate_forward(32)
    backward_time = profiler.estimate_backward(32)
    update_time = profiler.estimate_update(kbytes)
    total_time_seconds = (forward_time + backward_time + update_time)*0.001
    print("Estimated one batch gradient updated time: {} seconds".format(total_time_seconds))

def main():
    print("training sciprt starts....")
    #sys_argv[1] = first year for trianing data
    val_year = [2021]
    test_year = [2022]
    train_years = list(range(int(sys.argv[1]),2021))
    WIND_SIZE = int(sys.argv[2])
    BATCH_ITERATIONS = int(sys.argv[3])
    device_name = sys.argv[4]

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

    #print(train_set.shape)
    #print(val_set.shape)
    #print(test_set.shape)

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

    elapsed_time = 0
    iterations = BATCH_ITERATIONS

    for example_inputs, example_labels in window.train.take(1):
        #print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        #print(f'Labels shape (batch, time, features): {example_labels.shape}')
        one_batch_inputs = example_inputs
        one_batch_labels = example_labels

    print("Training started, running {0} iterations".format(iterations))
    for i in range(iterations):
        start = timer()
        model.train_on_batch(one_batch_inputs, one_batch_labels)
        end = timer()
        elapsed_time += timedelta(seconds=end-start).total_seconds()

    print("Average time to run single gradient update: {} seconds".format(elapsed_time/iterations))
    
    estimate_required_time(device_name)

if __name__ == "__main__":
    main()
