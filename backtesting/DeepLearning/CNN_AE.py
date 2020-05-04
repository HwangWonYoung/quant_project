# import packages
import sys
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# import dataset
KOSPI = pd.read_csv('KOSPI_new.csv')

# pre-defined functions
def TS_AE_dataformat(ts, length):
    
    # return X and Y using for loop
    X = []
    Y = []
    
    for i in range(len(ts)-length):
        X.append(np.array(ts[i:(i+length)]))
        Y.append(np.array(ts[i:(i+length)]))
            
    return np.array(X), np.array(Y)

def next_batch(x, y, batch_size):
    N = x.shape[0]
    batch_indices = np.random.permutation(N)[:batch_size]
    x_batch = x[batch_indices]
    y_batch = y[batch_indices]
    return x_batch, y_batch

def batch_normalization(batch):
    normalized_batch = (batch - np.mean(batch)) / np.std(batch)
    return normalized_batch

# 16 일 단위로 denoise
train_X, train_Y = TS_AE_dataformat(KOSPI['Close'], 16)

train_X = train_X.reshape(-1,16,1)
train_Y = train_Y.reshape(-1,16,1)

# reset grapth
tf.reset_default_graph()

# hyperparameters
num_epoch = 30
batch_size = 1
learning_rate = 0.01
display_step = 1

# place holder for input
X = tf.placeholder(tf.float32, [batch_size, 16, 1])
Y = tf.placeholder(tf.float32, [batch_size, 16, 1])

# CNN AutoEncoder
def CNN_AutoEncoder_for_TS(X):
    
    # upsampling 에 사용되는 filter & 마지막 층 출력에 사용되는 layer
    layer={'filter_1':tf.Variable(tf.random_normal([3, 32, 32])),
           'filter_2':tf.Variable(tf.random_normal([3, 32, 32]))}
    
    # encoding
    conv1 = tf.layers.conv1d(X, filters=32, kernel_size=3, strides=1, padding='SAME', activation='relu', use_bias=True, name='conv1') # 16 x 32
    maxpool1 = tf.layers.max_pooling1d(conv1, pool_size=2, strides=2, name='pool1') # 16 x 32
    conv2 = tf.layers.conv1d(maxpool1, filters=32, kernel_size=3, strides=1, padding='SAME', activation='relu', use_bias=True, name='conv2') # 8 x 32
    encoded = tf.layers.max_pooling1d(conv2, pool_size=2, strides=2, name='encoding') # 4 x 32
    
    # decoding
    conv3 = tf.layers.conv1d(encoded, filters=32, kernel_size=3, strides=1, padding='SAME', activation='relu', use_bias=True, name='conv3') # 4 x 32
    upsample1 = tf.nn.conv1d_transpose(conv3, filters=layer['filter_1'], output_shape=(batch_size, 8, 32), padding='SAME', strides=2, name='upsample1') # 8 x 32
    upsample2 = tf.nn.conv1d_transpose(upsample1, filters=layer['filter_2'], output_shape=(batch_size, 16, 32), padding='SAME', strides=2, name='upsample2') # 16 x 32
    
    logits = tf.layers.conv1d(upsample2,filters=1,kernel_size=3,strides=1,name='logits', padding='SAME', use_bias=True) # 16 x 1
    
    return logits

predicted_value = CNN_AutoEncoder_for_TS(X)
loss = tf.reduce_mean(tf.square(predicted_value - Y))
train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

# saver for modeling saving
saver = tf.train.Saver()
save_file = './train_model.ckpt'

train_loss_list = []

# model training
with tf.Session() as sess:
    # initialize variables
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(num_epoch):
        
        average_loss = 0
        total_batch = int(len(train_X)/batch_size)
        
        for i in range(total_batch):
            X_batch, Y_batch = next_batch(train_X, train_Y, batch_size)
            X_batch, Y_batch = batch_normalization(X_batch), batch_normalization(Y_batch)
            _, current_loss = sess.run([train_step,loss],feed_dict = {X: X_batch, Y: Y_batch})
            average_loss += current_loss/total_batch
        
        if epoch % display_step == 0:
            print('반복(Epoch): {}, 손실함수(Training Loss): {}'.format(epoch+1, average_loss))
            
            train_loss_list.append(average_loss)
            
    # Save the model
    saver.save(sess, save_file)
    print('Trained Model Saved.')
    
@ 아직 시도해 봐야 할 것 매우 많음...