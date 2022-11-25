import numpy as np
import pandas as pd
import tensorflow as tf

MAX_EPOCHS = 1

def compile_and_fit(model, window, optimizer, max_epochs = MAX_EPOCHS):
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience = 10, restore_best_weights = True)

    model.compile(loss=tf.losses.MeanSquaredError(),optimizer=optimizer,metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs,validation_data=window.val, callbacks=[early_stopping])
    return history
