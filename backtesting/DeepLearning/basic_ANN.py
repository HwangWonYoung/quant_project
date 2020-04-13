import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

# read training dataset
data = pd.read_csv('training_set_3y.csv')

# remove NAs
high_NA_columns = data.columns[data.apply(lambda x: sum(x.isna()), axis=0)/len(data) > 0.2]
data = data.drop(list(high_NA_columns), axis = 1)
data.dropna(inplace=True)

# binary Y
data.loc[:,'Y_1_month'] = data.Y_1_month.apply(lambda x: 1 if x > 0 else 0)

# basic ANN model
# only one layer

train = data.iloc[np.random.choice(len(data), 32000, replace=False),:]
test = data.iloc[-np.random.choice(len(data), 32000, replace=False),:]
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

def dense_to_one_hot(labels_dense, num_classes):
    """function converting labels into one hot encoded vector"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

X_train, Y_train = np.array(train.drop(['Y_1_month'], axis=1)), dense_to_one_hot(train['Y_1_month'])
X_test, Y_test = np.array(test.drop(['Y_1_month'], axis=1)), dense_to_one_hot(test['Y_1_month'])

# hyperparameters
learning_rate = 0.01
num_epochs = 30
batch_size = 1000
input_size = 55
hidden1_size = 100
output_size = 2
display_step = 1

x = tf.placeholder(tf.float32, shape = [None,input_size])
y = tf.placeholder(tf.float32, shape = [None, output_size])

def build_ANN(x):
    #Layer1
    W1 = tf.Variable(tf.random_normal(shape = [input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape = [hidden1_size]))
    H1_output = tf.nn.relu(tf.matmul(x,W1)+b1)
    #Layer 2
    W_output = tf.Variable(tf.random_normal(shape = [hidden1_size, output_size]))
    b_output = tf.Variable(tf.random_normal(shape = [output_size]))
    logits = tf.matmul(H1_output,W_output)+b_output
    return logits

predicted_value = build_ANN(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_value, labels=y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    # initialize variables
    sess.run(tf.global_variables_initializer())
    
    # 30 epochs
    for epoch in range(num_epochs):
        average_loss = 0
        total_batch = int(len(X_train)/batch_size)
        # 32 mini batches
        for i in range(total_batch):
            batch_x, batch_y = X_train[batch_size*(i):(batch_size*(i+1)-1)], Y_train[batch_size*(i):(batch_size*(i+1)-1)]
            _, current_loss = sess.run([train_step,loss],feed_dict = {x: batch_x, y: batch_y})
            # calculate average loss
            average_loss += current_loss/total_batch
            # print result per epoch
        if epoch % display_step == 0:
            print('반복(Epoch): {}, 손실함수(Loss): {}'.format(epoch+1, average_loss))
    #Model 성능 Test
    correct_prediction = tf.equal(tf.arg_max(predicted_value,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    print('정확도(Accuracy): %f'%(accuracy.eval(feed_dict={x:X_test, y: Y_test})))
    
@ 텐서플로우로 처음 구현해본 모델
@ layer, neuron, optimizer, batch size, epoch 등등 customize 할 수 있는게 굉장히 많다
@ 모델을 low level에서 다양화 할 수 있다는 것은 좋아 보이기도 하지만
@ 한편으로 어떤 식으로 모델을 구성해야 할 지 감이 잘 오지 않는다..ㅠ