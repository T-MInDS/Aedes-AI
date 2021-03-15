import tensorflow as tf

def ff_model(input_shape):
    xin = tf.keras.layers.Input(input_shape)
    
    c1 = tf.keras.layers.Conv1D(64, 3, activation = 'relu', data_format = 'channels_last')(xin)
    c2 = tf.keras.layers.Conv1D(64, 3, activation = 'relu', data_format = 'channels_last')(c1)

    h1 = tf.keras.layers.BatchNormalization()(c2)
    h2 = tf.keras.layers.Flatten()(h1)
    h3 = tf.keras.layers.Dense(64, activation = 'relu')(h2)
    h4 = tf.keras.layers.Dense(64, activation = 'relu')(h3)

    xout = tf.keras.layers.Dense(1, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001))(h4)
    
    return tf.keras.models.Model(xin, xout)

def lstm_model(input_shape):
    xin = tf.keras.layers.Input(input_shape)
    
    c1 = tf.keras.layers.Conv1D(64, 3, activation = 'relu', data_format = 'channels_last')(xin)
    c2 = tf.keras.layers.Conv1D(64, 3, activation = 'relu', data_format = 'channels_last')(c1)

    h1 = tf.keras.layers.BatchNormalization()(c2)
    
    r1 = tf.keras.layers.LSTM(64, return_sequences = True)(h1)
    r2 = tf.keras.layers.LSTM(64)(r1)

    xout = tf.keras.layers.Dense(1, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001))(r2)
    
    return tf.keras.models.Model(xin, xout)

def gru_model(input_shape):
    xin = tf.keras.layers.Input(input_shape)
    
    c1 = tf.keras.layers.Conv1D(64, 3, activation = 'relu', data_format = 'channels_last')(xin)
    c2 = tf.keras.layers.Conv1D(64, 3, activation = 'relu', data_format = 'channels_last')(c1)
    
    h1 = tf.keras.layers.BatchNormalization()(c2)
    
    r1 = tf.keras.layers.GRU(64, return_sequences = True)(h1)
    r2 = tf.keras.layers.GRU(64)(r1)

    xout = tf.keras.layers.Dense(1, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001))(r2)
    
    return tf.keras.models.Model(xin, xout)