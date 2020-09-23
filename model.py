import tensorflow as tf
import numpy as np
import pandas as pd
import time
from tensorflow.keras.callbacks import TensorBoard
from preprocessing_module import pre_process


def get_model(drop_prob, batch_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=135,kernel_initializer='normal' ,activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units=256,kernel_initializer='normal' ,activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units=256,kernel_initializer='normal' ,activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units=256,kernel_initializer='normal' ,activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(drop_prob))
    model.add(tf.keras.layers.Dense(units=1,kernel_initializer='normal' ,activation=tf.keras.activations.linear))
    return model


def get_dataset(path, test_ratio, batch_size):
    raw_data = pd.read_csv(path)

    x_data = raw_data.iloc[:, 1:-1]
    y_data = raw_data.iloc[:, -1]

    x_train, x_val, y_train, y_val = pre_process(x_data, y_data, test_ratio)

    x_train = np.expand_dims(x_train,axis=-1)
    y_train = np.expand_dims(y_train, axis=1)
    x_val = np.expand_dims(x_val,axis=-1)
    y_val = np.expand_dims(y_val, axis=1)

    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    def reshape(x,y):
        return x,y

    train__dataset = train_dataset.map(reshape).cache().shuffle(32).repeat(1).batch(batch_size)
    validation_dataset = validation_dataset.map(reshape).cache().shuffle(32).repeat(1).batch(batch_size)

    print(train__dataset)
    print(validation_dataset)

    return train_dataset, validation_dataset

def main():
    path = "./data/train.csv"
    epochs = 500
    batch_size = 32
    drop_prob = 0.25
    log_name = f"testing-{int(time.time())}"

    callback = TensorBoard(log_dir=f".\log\keras\{log_name}")
    train_dataset, validation_dataset = get_dataset(path, 0.1, batch_size)

    model = get_model(drop_prob, batch_size)

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    model.fit(train_dataset, epochs=epochs, verbose=2, callbacks=[callback], validation_data=validation_dataset)


if __name__ == "__main__":
    main()
