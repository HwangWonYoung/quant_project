# FFNN architecture
# Refer to "Deep Learning for Forecasting Stock Returns in the Cross-Section" by Masaya Abe1 and Hideki Nakayama2
# additional options : activation function, epoch numbers, learning rate

def build_NN3(x):
    """ NN3_1 : hidden1_size = 70, dropout = 0
        NN3_2 : hidden1_size = 80, dropout = 0 
        NN3_3 : hidden1_size = 100, dropout = 0
        NN3_4 : hidden1_size = 120, dropout = 0
        NN3_DO_1 : hidden1_size = 244, dropout = 0.5 
        NN3_DO_2 : hidden1_size = 322, dropout = 0.5
        NN3_DO_3 : hidden1_size = 354, dropout = 0.5
        NN3_DO_4 : hidden1_size = 399, dropout = 0.5 """
    #Layer1
    W1 = tf.Variable(tf.random_normal(shape = [input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape = [hidden1_size]))
    H1_output = tf.nn.relu(tf.matmul(x,W1)+b1)
    
    #Layer2
    W_output = tf.Variable(tf.random_normal(shape = [hidden1_size, output_size]))
    b_output = tf.Variable(tf.random_normal(shape = [output_size]))
    logits = tf.matmul(H1_output,W_output)+b_output
    
    return logits

def build_DNN5(x):
    """ DNN5_1 : hidden1_size = 100, hidden2_size = 50, hidden3_size = 10, droupout = 0.5
        DNN5_2 : hidden1_size = 100, hidden2_size = 70, hidden3_size = 50, droupout = 0.5
        DNN5_3 : hidden1_size = 120, hidden2_size = 70, hidden3_size = 20, droupout = 0.5 
        DNN5_4 : hidden1_size = 120, hidden2_size = 80, hidden3_size = 40, droupout = 0.5 """
    #Layer1
    W1 = tf.Variable(tf.random_normal(shape = [input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape = [hidden1_size]))
    H1_output = tf.nn.relu(tf.matmul(x,W1)+b1)
    
    #Layer2
    W2 = tf.Variable(tf.random_normal(shape = [hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random_normal(shape = [hidden2_size]))
    H2_output = tf.nn.relu(tf.matmul(H1_output,W2)+b2)
    
    #Layer3
    W3 = tf.Variable(tf.random_normal(shape = [hidden2_size, hidden3_size]))
    b3 = tf.Variable(tf.random_normal(shape = [hidden3_size]))
    H3_output = tf.nn.relu(tf.matmul(H2_output,W3)+b3)
    
    #Layer4
    W_output = tf.Variable(tf.random_normal(shape = [hidden3_size, output_size]))
    b_output = tf.Variable(tf.random_normal(shape = [output_size]))
    logits = tf.matmul(H3_output,W_output)+b_output
    
    return logits

def build_DNN8(x):
    """ DNN8_1 : hidden1_size = 100, hidden2_size = 100, hidden3_size = 50, hidden4_size = 50, hidden5_size = 10, hidden6_size = 10, droupout = 0.5
        DNN8_1 : hidden1_size = 100, hidden2_size = 100, hidden3_size = 70, hidden4_size = 70, hidden5_size = 50, hidden6_size = 50, droupout = 0.5
        DNN8_1 : hidden1_size = 120, hidden2_size = 120, hidden3_size = 70, hidden4_size = 70, hidden5_size = 20, hidden6_size = 20, droupout = 0.5
        DNN8_1 : hidden1_size = 120, hidden2_size = 120, hidden3_size = 80, hidden4_size = 80, hidden5_size = 40, hidden6_size = 40, droupout = 0.5 """
    #Layer1
    W1 = tf.Variable(tf.random_normal(shape = [input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape = [hidden1_size]))
    H1_output = tf.nn.relu(tf.matmul(x,W1)+b1)
    
    #Layer2
    W2 = tf.Variable(tf.random_normal(shape = [hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random_normal(shape = [hidden2_size]))
    H2_output = tf.nn.relu(tf.matmul(H1_output,W2)+b2)
    
    #Layer3
    W3 = tf.Variable(tf.random_normal(shape = [hidden2_size, hidden3_size]))
    b3 = tf.Variable(tf.random_normal(shape = [hidden3_size]))
    H3_output = tf.nn.relu(tf.matmul(H2_output,W3)+b3)
    
    #Layer4
    W4 = tf.Variable(tf.random_normal(shape = [hidden3_size, hidden4_size]))
    b4 = tf.Variable(tf.random_normal(shape = [hidden4_size]))
    H4_output = tf.nn.relu(tf.matmul(H3_output,W4)+b4)
    
    #Layer5
    W5 = tf.Variable(tf.random_normal(shape = [hidden4_size, hidden5_size]))
    b5 = tf.Variable(tf.random_normal(shape = [hidden5_size]))
    H5_output = tf.nn.relu(tf.matmul(H4_output,W5)+b5)
    
    #Layer6
    W6 = tf.Variable(tf.random_normal(shape = [hidden5_size, hidden6_size]))
    b6 = tf.Variable(tf.random_normal(shape = [hidden6_size]))
    H6_output = tf.nn.relu(tf.matmul(H5_output,W6)+b6)
    
    #Layer7
    W_output = tf.Variable(tf.random_normal(shape = [hidden6_size, output_size]))
    b_output = tf.Variable(tf.random_normal(shape = [output_size]))
    logits = tf.matmul(H6_output,W_output)+b_output
    
    return logits