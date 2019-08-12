# -*- coding: utf-8 -*- 
#############################################  Tensorflow ############################################
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import initializers
from keras.utils import to_categorical


BANTCH_SIZE = 64
VOCAB_SIZE = 48849
STEP_NUM = 5572       # word length
WORD_LENTH = 5572
EMBEDDING_DIM = 50     # word embedding will be 2 dimension for 2d visualization
CLASS_NUM = 2
UNITS_NUM = 100
PROJECTION_NUM = 64
OUTPUT_KEEP_PROB = 0.5
LEARNING_RATE = 0.001
EPOCHS_NUM = 10
DROP_KEEP_PROB = 0.5


def lstm_model(embed):

    # making placeholders for X_train and Y_train
    # 声明训练数据的变量以及softmax矩阵的变量，其中input_data和target的维度就是train_data和train_label的第一维度其实是小于等于batch_size的（因为sample_num ％ batch_size不一定等于0)，所以写成None
    input_data = tf.placeholder(tf.int64, [None, STEP_NUM], name="input_data")
    target = tf.placeholder(tf.int64, [None,CLASS_NUM], name="target_label") 
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    #length = tf.placeholder(tf.float32, [None], name='rnn_length')

    l2_loss = tf.constant(0.0)
    #target.shape
    # cast our label to float32
    target = tf.cast(target,tf.float32)


    # Embedding layer  
    #embedding = tf.Variable(name = 'embedding', shape=embed.shape, initializer=tf.constant_initializer(embed), trainable=False)  ## trainable=True
    embedding = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_DIM], -1.0, 1.0), name="W_text")
    input_embedding = tf.nn.embedding_lookup(embedding, input_data)   #得到与输入 input_data 对应的词向量,最后 input_embedding的维度为[BANTCH_SIZE, STEP_NUM, EMBEDDING_DIM]   None X 5572 X 50  通过input_embedding.shape来查看
    #input_embedding = tf.unstack(input_embedding_, STEP_NUM, 1)



    # 建立lstm节点
    # build lstm network


    #lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(
            #num_units=UNITS_NUM,
            #forget_bias=0.3,
            #initializer=initializers.xavier_initializer(),
            #use_peepholes=False,
            #num_proj=PROJECTION_NUM,
           # name='LSTM_CELL_1')
    
    lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(UNITS_NUM)
   # lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(UNITS_NUM)
   # lstm_cell_3 = tf.nn.rnn_cell.BasicLSTMCell(UNITS_NUM)
    
    lstm_cell_1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_1, output_keep_prob= DROP_KEEP_PROB)
    #lstm_cell_2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_2, output_keep_prob= DROP_KEEP_PROB)
    #lstm_cell_3 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_3, output_keep_prob= DROP_KEEP_PROB)
    #lstm_cell_2 = tf.contrib.rnn.LSTMCell(
    #        num_units=UNITS_NUM,
            #forget_bias=0.3,
    #        use_peepholes=False,
    #        initializer=initializers.xavier_initializer(),
    #        num_proj=PROJECTION_NUM,
    #        name='LSTM_CELL_2')




   # stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1,lstm_cell_2,lstm_cell_3])
    #stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1])



    # Extrqact the bacth size - this allows for variable batch size
    #current_batch_size = tf.shape(input_data)[0]

    # Create LSTM Initial State of Zeros 后续的学习中一点一点更新 State
    #initial_state = lstm_cell_1.zero_state(current_batch_size, tf.float32)

    # Wrap our lstm cell in a dropout Wrapper
    #stacked_lstm = tf.contrib.rnn.DropoutWrapper(cell = stacked_lstm,
    #                                         output_keep_prob = OUTPUT_KEEP_PROB)



    #relevant = tf.sign(tf.abs(input_data))                       # [[1. 1... 1. 1. 1.] , [1. 1... 1. 1. 1.] , [1. 1... 1. 1. 1.]]
    #length = tf.reduce_sum(relevant, reduction_indices=1)        # [5572, 5572, 5572, ..., 5572]
    #text_length = tf.cast(length, tf.float32)                    # [5572, 5572, 5572, ..., 5572]

    # 根据之前建立的lstm节点建立rnn网络，返回值为outputs和state 
    # outputs维度通过outputs.shape来查看后为 [BANTCH_SIZE, STEP_NUM , UNITS_NUM], 是最后一层每个 STEP 的输出   None X 5572 X 128
    # outputs中包含5572是因为包含了5572个timestep所有的输出。因此后续只需要提取最后一个时刻的即可
    # final_state是每行最后一个tensor的集合，维度为[BANTCH_SIZE, UNITS_NUM]
    all_outputs, final_state = tf.nn.dynamic_rnn( cell = lstm_cell_1, 
                                                  inputs = input_embedding, 
                                                  #initial_state = initial_state,
                                                  #sequence_length= text_length,
                                                  dtype = tf.float32)
    
    
    
    #batch_size = tf.shape(input_data)[0]                                  # input_data shape is [None, STEP_NUM]  
    #max_length = int(all_outputs.get_shape()[1])                          # all_outputs shape is None X 5572 X 100
    #input_size = int(all_outputs.get_shape()[2])                          # all_outputs shape is None X 5572 X 100
    #index = tf.range(0, batch_size) * max_length + (text_length - 1)
    #flat = tf.reshape(all_outputs, [-1, input_size])
    #h_outputs = tf.gather(flat, index)



    

    #outputs, final_state = rnn.static_rnn(lstm_cell_1,input_embedding,dtype=tf.float32)  # (28 X None X 28) X (28 X 128) -> 28 X None X 128
    # instantiate weights and biases
    # softmax_w和softmax_b，就是接在hidden_layer之后的全连接层；它下接hidden_layer，上接pred_class
    # softmax_w = tf.Variable(tf.truncated_normal( [PROJECTION_NUM, CLASS_NUM],mean=0.0,stddev=0.001), dtype=tf.float32)  # 生成 mean=0，std=1随机值   维度：128 X 1
    softmax_w = tf.get_variable("softmax_w", shape=[UNITS_NUM, CLASS_NUM] ,initializer=tf.contrib.layers.xavier_initializer()) 

    softmax_b = tf.Variable(tf.constant(0.1, shape=[CLASS_NUM]), name="softmax_b", dtype=tf.float32)   # 维度：2

    l2_loss += tf.nn.l2_loss(softmax_w)
    
    l2_loss += tf.nn.l2_loss(softmax_b)
            
            
            
            
    # 将最后一个时刻的状态（其实就是叠加到最后的一个STEP_NUM）做softmax得到预测结果。由于上一步骤的outputs维度为 [BANTCH_SIZE, STEP_NUM , UNITS_NUM]，所以需要先做一个矩阵变换, 即 None X 5572 X 128 --> 5572 X None X 128
    all_outputs = tf.transpose(all_outputs,[1,0,2])  # 5572 X None X 128

    prediction = tf.nn.xw_plus_b(all_outputs[-1], softmax_w, softmax_b) # outputs[-1] 为最后一个时刻的输出: [None,128]. 因此 results = None X 128 X 128 X 2 + 2 = None X 2 + 2 = None X 2
    prediction_prob = tf.nn.softmax (prediction)


    # 逐个元素进行判断，如果相等就是True，不相等，就是False
    # predictions - [1,1,0,0]
    # labels - [1,0,0,1]
    # correct_prediction = [1,0,1,0]
    #correct_prediction = tf.equal(tf.to_int64(prediction_prob > 0.5),tf.argmax(target,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.equal(tf.argmax(tf.nn.sigmoid(prediction),axis=1),tf.argmax(target,axis=1))


    #AUC
    #target_one_hot = tf.one_hot(tf.cast(target, tf.int32), CLASS_NUM )
    #auc, auc_update_op = tf.metrics.auc(target,prediction)
    # target:[None,2]    prediction:[None,2]


    # Choice our model made
    #choice = tf.argmax(tf.nn.sigmoid(prediction),axis=1)

    # Calculate the loss given prediction and labels
    # 计算softmax(logits) 和 labels 之间的交叉熵
    # labels --> 真实数据的类别标签. softmax_cross_entropy_with_logits_v2要求labels是一个数值，这个数值记录着ground truth所在的索引。以[0,0,1,0]为例，这里真值1的索引为2。
    #                             所以softmax_cross_entropy_with_logits_v2要求labels的输入为数字2(tensor)。一般可以用tf.argmax()来从[0,0,1,0]中取得真值的索引。
    # logits --> 神经网络最后一层的类别预测输出值 (直接由神经网络输出的数值, 比如[3.5, 2.1, 7.89, 4.4] )

    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction,labels = target))   #交叉熵损失函数
    #loss = tf.reduce_mean(tf.nn.softmax(logits = prediction))   #交叉熵损失函数
    loss= tf.keras.losses.categorical_crossentropy( target, tf.nn.softmax(prediction), from_logits=False)

    #losses = tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels= target)
    #loss = tf.reduce_mean(losses) 
            
    # Declare our optimizer, in this case RMS Prop
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    #optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    saver = tf.train.Saver()


    return BANTCH_SIZE, optimizer, loss, accuracy, input_data, target, prediction, all_outputs, softmax_w, softmax_b, embedding, prediction_prob



