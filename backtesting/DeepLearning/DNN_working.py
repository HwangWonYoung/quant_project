import os
import sys
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from itertools import compress
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

path = os.getcwd()
datasets = glob.glob(path + "/*.csv")

#  사용할 변수 미리 지정하자
pos_vars = ["ROA_1", "ROE_1", "CFO", "ACCURUAL_1", 
              "LIQUIDITY", "MARGIN", "OP_MARGIN", "TURNOVER", "GPA", "ROC", "ROIC",
              "sales_growth_QoQ", "op_growth_QoQ", "net_growth_QoQ", "sales_growth_YoY",
              "op_growth_YoY", "net_growth_YoY", "sales_growth_3YoY", "op_growth_3YoY",
              "net_growth_3YoY", "op_turn_profit", "net_turn_profit", "EY_r", "OEY_r",
              "CFY_r", "idiosyncratic_momentum_6_2", "idiosyncratic_momentum_12_2",
              "momentum_6_2", "momentum_12_2", "beta_one_year", "beta_two_year"]

neg_vars = ["DE", "op_turn_loss", "net_turn_loss", "PBR_r", "PSR_r",
              "idiosyncratic_daily_vol_oneyear", "idiosyncratic_weekly_vol_oneyear",
              "daily_vol_oneyear", "daily_vol_twoyear", "weekly_vol_oneyear", "weekly_vol_twoyear",
              "momentum_1", "MDD"]

Y_vars = ["Y_2_week", "Y_1_month", "Y_2_month", "Y_3_month", "Y_6_month", "Y_12_month",
            "sharpe_ratio_1m_Y", "sharpe_ratio_3m_Y", "sharpe_ratio_6m_Y"]

etc_vars = ["name", "code", "market", "sector", "price", "market_cap", "size_level"]

X_columns = ["ROA_1", "ROE_1", "CFO", "ACCURUAL_1", 
              "LIQUIDITY", "MARGIN", "OP_MARGIN", "TURNOVER", "GPA", "ROC", "ROIC",
              "sales_growth_QoQ", "op_growth_QoQ", "net_growth_QoQ", "sales_growth_YoY",
              "op_growth_YoY", "net_growth_YoY", "sales_growth_3YoY", "op_growth_3YoY",
              "net_growth_3YoY", "op_turn_profit", "net_turn_profit", "EY_r", "OEY_r",
              "CFY_r", "idiosyncratic_momentum_6_2", "idiosyncratic_momentum_12_2",
              "momentum_6_2", "momentum_12_2", "beta_one_year", "beta_two_year", "DE", "op_turn_loss", "net_turn_loss", "PBR_r", "PSR_r",
              "idiosyncratic_daily_vol_oneyear", "idiosyncratic_weekly_vol_oneyear",
              "daily_vol_oneyear", "daily_vol_twoyear", "weekly_vol_oneyear", "weekly_vol_twoyear",
              "momentum_1", "MDD"]

Y_column = "Y_1_month"

# 'sector' 변수는 어떤 식으로 encoding 하는 것이 제일 좋을지 고민해보고 나중에 모델링에 추가하도록 하자
# embedding 에 대해 공부해보고 진행해야 될 듯 싶다

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def load_train_test(filenames, startpoint, X_columns, Y_column , num_of_trainsets = 20, num_of_testsets = 1):
    
    """ filenames : load 할 dataset
         startpoint : filenames 중 어디서부터 load 할지를 표시, 나중에 반복적으로 train & test set 을 load 하기 위해 필요함 """
    
    trainset_names = filenames[startpoint:(startpoint+num_of_trainsets)]
    testset_names = filenames[(startpoint+num_of_trainsets):(startpoint+num_of_trainsets+num_of_testsets)]
    
    train_list = []
    test_list = []
    
    for name in trainset_names:
        temp_train = pd.read_csv(name, encoding = 'euc-kr')
        temp_train = temp_train[[Y_column]+X_columns].dropna()
        temp_train.loc[:,Y_column] = temp_train[Y_column].apply(lambda x: 1 if x > 0 else 0)
        train_list.append(temp_train)
    
    train = pd.concat(train_list)
    
    for name in testset_names:
        temp_test = pd.read_csv(name, encoding = 'euc-kr')
        temp_test = temp_test[[Y_column]+X_columns].dropna()
        temp_test.loc[:,Y_column] = temp_test[Y_column].apply(lambda x: 1 if x > 0 else 0)
        test_list.append(temp_test)
        
    test = pd.concat(test_list)
    
    X_train, Y_train = np.array(train.drop([Y_column], axis=1)), dense_to_one_hot(train[Y_column], 2)
    X_test, Y_test = np.array(test.drop([Y_column], axis=1)), dense_to_one_hot(test[Y_column], 2)
    
    return X_train, Y_train, X_test, Y_test

# 데이터 셋 로드
# startpoint 를 looping 하면서 rolling window 할 수 있음
# 모델의 최종적인 성능 확인을 위해 나중엔 rolling window를 사용하자
X_train, Y_train, X_test, Y_test = load_train_test(datasets, 0, X_columns = X_columns, Y_column = Y_column , num_of_trainsets = 30, num_of_testsets = 1)

#################################################################### Model Candidates##################################################################
def build_NN3(x):
    """ NN3_1 : hidden1_size = 70, keep_prob = 0
        NN3_2 : hidden1_size = 80, keep_prob = 0 
        NN3_3 : hidden1_size = 100, keep_prob = 0
        NN3_4 : hidden1_size = 120, keep_prob = 0
        NN3_DO_1 : hidden1_size = 244, keep_prob = 0.5 
        NN3_DO_2 : hidden1_size = 322, keep_prob = 0.5
        NN3_DO_3 : hidden1_size = 354, keep_prob = 0.5
        NN3_DO_4 : hidden1_size = 399, keep_prob = 0.5 """
    #Layer1
    W1 = tf.Variable(tf.random_normal(shape = [input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape = [hidden1_size]))
    H1_output = tf.nn.tanh(tf.matmul(x,W1)+b1)
    H1_output = tf.nn.dropout(H1_output, keep_prob)
    
    #Layer2
    W_output = tf.Variable(tf.random_normal(shape = [hidden1_size, output_size]))
    b_output = tf.Variable(tf.random_normal(shape = [output_size]))
    logits = tf.matmul(H1_output,W_output)+b_output
    
    return logits

def build_DNN5(x):
    """ DNN5_1 : hidden1_size = 100, hidden2_size = 50, hidden3_size = 10, keep_prob = 0.5
        DNN5_2 : hidden1_size = 100, hidden2_size = 70, hidden3_size = 50, keep_prob = 0.5
        DNN5_3 : hidden1_size = 120, hidden2_size = 70, hidden3_size = 20, keep_prob = 0.5 
        DNN5_4 : hidden1_size = 120, hidden2_size = 80, hidden3_size = 40, keep_prob = 0.5 """
    #Layer1
    W1 = tf.Variable(tf.random_normal(shape = [input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape = [hidden1_size]))
    H1_output = tf.nn.tanh(tf.matmul(x,W1)+b1)
    H1_output = tf.nn.dropout(H1_output, keep_prob)
    
    #Layer2
    W2 = tf.Variable(tf.random_normal(shape = [hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random_normal(shape = [hidden2_size]))
    H2_output = tf.nn.tanh(tf.matmul(H1_output,W2)+b2)
    H2_output = tf.nn.dropout(H2_output, keep_prob)
    
    #Layer3
    W3 = tf.Variable(tf.random_normal(shape = [hidden2_size, hidden3_size]))
    b3 = tf.Variable(tf.random_normal(shape = [hidden3_size]))
    H3_output = tf.nn.tanh(tf.matmul(H2_output,W3)+b3)
    H3_output = tf.nn.dropout(H3_output, keep_prob)
    
    #Layer4
    W_output = tf.Variable(tf.random_normal(shape = [hidden3_size, output_size]))
    b_output = tf.Variable(tf.random_normal(shape = [output_size]))
    logits = tf.matmul(H3_output,W_output)+b_output
    
    return logits

def build_DNN8(x):
    """ DNN8_1 : hidden1_size = 100, hidden2_size = 100, hidden3_size = 50, hidden4_size = 50, hidden5_size = 10, hidden6_size = 10, keep_prob = 0.5
        DNN8_1 : hidden1_size = 100, hidden2_size = 100, hidden3_size = 70, hidden4_size = 70, hidden5_size = 50, hidden6_size = 50, keep_prob = 0.5
        DNN8_1 : hidden1_size = 120, hidden2_size = 120, hidden3_size = 70, hidden4_size = 70, hidden5_size = 20, hidden6_size = 20, keep_prob = 0.5
        DNN8_1 : hidden1_size = 120, hidden2_size = 120, hidden3_size = 80, hidden4_size = 80, hidden5_size = 40, hidden6_size = 40, keep_prob = 0.5 """
    #Layer1
    W1 = tf.Variable(tf.random_normal(shape = [input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape = [hidden1_size]))
    H1_output = tf.nn.tanh(tf.matmul(x,W1)+b1)
    H1_output = tf.nn.dropout(H1_output, keep_prob)
    
    #Layer2
    W2 = tf.Variable(tf.random_normal(shape = [hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random_normal(shape = [hidden2_size]))
    H2_output = tf.nn.tanh(tf.matmul(H1_output,W2)+b2)
    H2_output = tf.nn.dropout(H2_output, keep_prob)
    
    #Layer3
    W3 = tf.Variable(tf.random_normal(shape = [hidden2_size, hidden3_size]))
    b3 = tf.Variable(tf.random_normal(shape = [hidden3_size]))
    H3_output = tf.nn.tanh(tf.matmul(H2_output,W3)+b3)
    H3_output = tf.nn.dropout(H3_output, keep_prob)
    
    #Layer4
    W4 = tf.Variable(tf.random_normal(shape = [hidden3_size, hidden4_size]))
    b4 = tf.Variable(tf.random_normal(shape = [hidden4_size]))
    H4_output = tf.nn.tanh(tf.matmul(H3_output,W4)+b4)
    H4_output = tf.nn.dropout(H4_output, keep_prob)
    
    #Layer5
    W5 = tf.Variable(tf.random_normal(shape = [hidden4_size, hidden5_size]))
    b5 = tf.Variable(tf.random_normal(shape = [hidden5_size]))
    H5_output = tf.nn.tanh(tf.matmul(H4_output,W5)+b5)
    H5_output = tf.nn.dropout(H5_output, keep_prob)
    
    #Layer6
    W6 = tf.Variable(tf.random_normal(shape = [hidden5_size, hidden6_size]))
    b6 = tf.Variable(tf.random_normal(shape = [hidden6_size]))
    H6_output = tf.nn.tanh(tf.matmul(H5_output,W6)+b6)
    H6_output = tf.nn.dropout(H6_output, keep_prob)
    
    #Layer7
    W_output = tf.Variable(tf.random_normal(shape = [hidden6_size, output_size]))
    b_output = tf.Variable(tf.random_normal(shape = [output_size]))
    logits = tf.matmul(H6_output,W_output)+b_output
    
    return logits
#################################################################### Model Candidates##################################################################

# hyperparameters
learning_rate = 0.05
num_epochs = 30
batch_size = 640
input_size = 44
hidden1_size = 100
hidden2_size = 70
hidden3_size = 50
# hidden4_size = 50
# hidden5_size = 10
# hidden6_size = 10
output_size = 2
display_step = 1

# placeholder
x = tf.placeholder(tf.float32, shape = [None,input_size])
y = tf.placeholder(tf.float32, shape = [None, output_size])
keep_prob = tf.placeholder(tf.float32)

predicted_value = build_DNN5(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_value, labels=y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# save loss history
train_loss_list = []
test_loss_list = []

# Session
with tf.Session() as sess:
    # initialize variables
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(num_epochs):
        average_loss = 0
        total_batch = int(len(X_train)/batch_size)
        
        # shuffle
        # shuffle_index = np.random.choice(len(X_train), len(X_train), replace = False)
        # X_train_shuffled = X_train[shuffle_index]
        # Y_train_shuffled = Y_train[shuffle_index]
        
        x_batches = np.array_split(X_train, total_batch) # 만약 shuffle 한다면 X_train 대신 X_train_shuffled 사용할 것
        y_batches = np.array_split(Y_train, total_batch) # 만약 shuffle 한다면 Y_train 대신 Y_train_shuffled 사용할 것

        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            _, current_loss = sess.run([train_step,loss],feed_dict = {x: batch_x, y: batch_y, keep_prob: 0.5})
            # calculate average loss
            average_loss += current_loss/total_batch
            # print result per epoch
        if epoch % display_step == 0:
            test_loss = sess.run([loss],feed_dict = {x: X_test, y: Y_test, keep_prob: 1})
            print('반복(Epoch): {}, 손실함수(Training Loss): {}'.format(epoch+1, average_loss))
            print('반복(Epoch): {}, 손실함수(Test Loss): {}'.format(epoch+1, test_loss))
            train_loss_list.append(average_loss)
            test_loss_list.append(test_loss)
            
    # Model 최종 성능 Test
    _, roc_score = tf.metrics.auc(y, tf.sigmoid(predicted_value))
    sess.run(tf.local_variables_initializer())
    print('AUC :', sess.run(roc_score, feed_dict={x : X_test, y : Y_test, keep_prob : 1 }))
    

@ 현재는 시간이 너무 오래 걸려서 window 를 이동하면서 모델의 성능을 테스트하는 작업은 하지 못했다
@ 코드를 미리 짜놓고 밤에 돌려봐야 할 것 같다
@ loss 가 어떻게 변화는지 살펴보니 epoch 극 초반에만 training loss가 준다
@ 금융 데이터에 딥러닝을 접목시키려면 shallow 하게 학습해야한다고 했는데 맞는 말인 건지 내가 잘못한건지 모르겠다
@ drop out은 크게 성능 향상을 가져오진 않았다
@ test loss 는 epoch 과는 무관하게 변화한다... 최종 auc도 별로 좋지 않다

@ 확인해 봐야 할 것 : learning rate 조절, 적절한 epoch 수 탐색, mini batch size, shuffling, normalization(중요!)