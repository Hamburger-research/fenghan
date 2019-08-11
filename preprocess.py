import argparse
import csv
import h5py
import re
import os

import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split

np.random.seed(1)


class Indexer:  #ex: clean_str(inputs[1]) 输入进来
    def __init__(self):
        self.counter = 2
        self.d = {"<unk>": 1} #初始化 Indexer 中的 dictionary时，如果在word2vec里面没有的，都填充成 1
        self.rev = {}
        self._lock = False

    def convert(self, w):
        if w not in self.d:  # 在不断填充 Indexer 中的 dictionary 的过程中， 如果是新单词（dictionary中没有），则需填充进来
            if self._lock:
                return self.d["<unk>"]
            self.d[w] = self.counter
            self.rev[self.counter] = w
            self.counter += 1
        return self.d[w]

    def lock(self):
        self._lock = True

    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.items()]
        items.sort()
        for v, k in items:
            print ( out, k, v )
        out.close()


def load_bin_vec(filename):
    """
    Loads a word2vec file and creates word2idx
    :param filename: The name of the file with word2vec vectors
    :return: word2vec dictionary, the size of embeddings and number of words in word2vec
    """
    w2v = {}
    with open(filename, 'r') as f:
        header = f.readline()  # 每一行每一行的读取
        vocab_size, emb_size = map(int, header.split()) # split() 以空格和\n为分隔符，将一行分成若干个字符串，存储在一个list里面
        for line in f:
            cline = line.split()
            w2v[cline[0]] = np.array(cline[1:], dtype=np.float64) # cline[0]就是每行的第一个字符(也就是单词). w2v 为 dictionary

    return w2v, emb_size, vocab_size



def parse_input_csv(filename, textfield, conditions, id_field, subj_field, chart_field):
    """
    Loads a CSV file and returns the texts as well as the condition-labels
    """
    texts = []
    target = []
    ids = []  # HAdmID
    subj = [] # subject id
    time = [] # chart time
    print("Parsing:", filename)
    with open(filename, 'r') as f:
        reader = csv.reader(f)  # , dialect=csv.excel_tab)
        field2id = {}
        for i, row in enumerate(reader):
            if i == 0:
                field2id = {fieldname: index for index, fieldname in enumerate(row)}
                print (field2id)
            else:
                texts.append("<padding> " * args.padding + row[field2id[textfield]] + " <padding>" * args.padding)
                current_targets = []
                for c in conditions:
                    current_targets.append(row[field2id[c]])
                target.append(current_targets)
                # store hospital admission ID
                # ids.append(row[field2id[id_field]])
                ids.append(i-1)
                subj.append(row[field2id[subj_field]])
                time.append(row[field2id[chart_field]])
    return texts, target, ids, subj, time


def clean_str(string):
    """
    Tokenization/string cleaning.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) # ^非
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()  # .lower() word2vec is case sensitive


# FILE_PATHS = [# 'nursingNotesClean.csv',
#               # 'dischargeSummariesClean.csv',
#                 'AllDischargeFinal24Oct16.csv']

args = {}


def main():
     
    os.chdir('/Users/FengHan/Desktop/deep learning')
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    #parser.add_argument('clean_summaries0209.csv', help="Source Input file", type=str)
    #parser.add_argument('word2vec_50d.txt', help="word2vec file", type=str)
    parser.add_argument('--padding', help="padding around each text", type=int, default=4)
    parser.add_argument('--batchsize', help="batchsize if you want to batch the data", type=int, default=1)
    parser.add_argument('--max_note_len', help="Cut off all notes longer than this (0 = no cutoff).", type=int,
                        default=0)
    parser.add_argument('--filename', help="File name for output file", type=str, default="data.h5")
    args = parser.parse_args()

    # LOAD THE WORD2VEC FILE
    word2vec, emb_size, v_large = load_bin_vec("word2vec_50d.txt") # word2vec 整个数据集（label+unlabeled）   470260
    
    
    
    

    print ('WORD2VEC POINTS:', v_large)

    # FIELDNAMES IN CSV FILE
    textfield = 'text'
    id_field = "Hospital.Admission.ID"
    subj_field = "subject.id"
    chart_field = "chart.time"
    conditions = ['cohort', #1
                  'Obesity', #2
                  'Non.Adherence', #3
                  'Developmental.Delay.Retardation', #4
                  'Advanced.Heart.Disease', #5
                  'Advanced.Lung.Disease', #6
                  'Schizophrenia.and.other.Psychiatric.Disorders', #7
                  'Alcohol.Abuse', #8
                  'Other.Substance.Abuse', #9
                  'Chronic.Pain.Fibromyalgia', #10
                  'Chronic.Neurological.Dystrophies', #11
                  'Advanced.Cancer', #12
                  'Depression', #13
                  'Dementia', #14
                  'Unsure']

    # LOAD ALL THE DATA INTO ARRAY

    inputs, targets, ids, subj, time = parse_input_csv("clean_summaries0209.csv", textfield, conditions, id_field, subj_field, chart_field)
    
    print ("FOUND {} DATA POINTS".format(len(inputs)))




    # CONVERT ALL THE TEXT  把 1610个inputs中的 text区域的 字符 全都变成 数字index
    lbl = []
    tokenizer = Indexer()  # Indexer 是开头定义的一个Class，tokenizer是一个object
    tokenizer.convert('padding')  # Indexer 是开头定义的一个Class，tokenizer是一个object
    max_len_sent = 0

    for i, t in enumerate(inputs): # 遍历 inputs list， i 是 index，t 是 elements
        current_convert = [tokenizer.convert(w) for w in clean_str(t).split()]  # 字符串转化成数字
        max_len_sent = max(max_len_sent, len(current_convert)) # 找出最长的 text 里 包含了多少个split出来的单词
        lbl.append(current_convert)
        if i % 100 == 0:
            print ("CONVERTING ROW {}".format(i))
    print ("MAXIMUM TEXT LENGTH IS {}".format(max_len_sent))
    print ("WORD AMOUNTS IS {}".format(tokenizer.counter))        
    
    

    # ADD PADDING TO GET TEXT INTO EQUAL LENGTH  让 全部1610个的病例中的字符向量长度相同, 使它们均有 5572 个字符index
    for sent in lbl:
        if len(sent) < max_len_sent:
            sent.extend([2] * (max_len_sent - len(sent)))
            
            
            
            
    # CUT OFF NOTE IF CUTOFF > 0.
    if args.max_note_len > 0:
        print("SHORTENING NOTES FROM {} TO {}".format(max_len_sent, args.max_note_len))
        max_len_sent = min(args.max_note_len, len(sent))
        lbl = [sent[:max_len_sent] for sent in lbl]
        
        
        

    # TAKING CARE OF DATA TYPE, PUT IDS WITH TEXT!  lbl 由 list 转化成 DataFrame, 并在最后 3 列添加 ids, subj, time 1610 X 5575
    # lbl 和 targets 都对应于 inputs （1610）
    lbl = np.column_stack([lbl, ids, subj, time])
    targets = np.array(targets, dtype=int)  # 标签也要转化成 DataFrame  1610 X 15
    print ("VOCAB SIZE {}".format(len(tokenizer.d))) # labele数据 1610个数据

    # remove the ids from the training texts 
    def split_input_id(data):
        return data[:, :-3], data[:, -3], data[:, -2], data[:, -1]

    # CONSTRUCT CORRECT EMBEDDING TABLE  
    embed = np.random.uniform(-0.25, 0.25, (len(tokenizer.d), emb_size))   # 44844 X 50 ; tokenizer 存储了 44844 个 来自 inputs 中的有效单词
    embed = np.row_stack((np.zeros(50),embed))
    embed[2,:] = 0
    
    embed[2,:] = 0
    
    
    
    unks = 0

    for key, value in tokenizer.d.items():  #  字典 tokenizer.d:  '<unk>': 1, 'padding': 2, 'Admission': 3, 'Date': 4 ...

        try:
            # -1 because of 1 indexing of word2idx (easier with torch)
            embed[value - 1] = word2vec[key]   # 例如： embed[2] <-- word2vec['Admission'] 
        except:
            unks += 1
            pass
    print ("{} UNKNOWN WORDS".format(unks))   

    # STORE ALL THE DATA   
    #tokenizer.write("words.dict")


    
#    with open("conditions.dict", 'w') as f:
#        for i, c in enumerate(conditions):
#            print (f, i + 1, c)

    filename = args.filename
 
    
################################### CROSS VALIDATION ###################################
    lbl_ = np.array(lbl, dtype=int)
    targets_ = np.array(targets, dtype=int)
    
    dataset_concatenate = np.concatenate((lbl_, targets_),axis=1)

    phenotypedictpos = dict({"Cancer":161,"Heart":275,"Lung":167,"Neuro":368,"Pain":321,"Alcohol":196,"Substance":155,"Obesity":126,"Disorders":295,"Depression":460})
    phenotypedict = dict({"Cancer":11,"Heart":4,"Lung":5,"Neuro":10,"Pain":9,"Alcohol":7,"Substance":8,"Obesity":1,"Disorders":6,"Depression":12})
    
    topred = 'Depression'

    DISEASE_ID = phenotypedict[topred]  # 第6种病， 论文中的 Psychiatric disorders
    

    pred = phenotypedict[topred]  #12
    
    pred_pos = phenotypedictpos[topred]
    
    targets_pred = targets[:,pred]
    targets_pred =  np.column_stack([targets_pred, ids, subj, time])
    targets_pred = targets_pred[targets_pred[:,0].argsort()]
    pos_sam = targets_pred[-pred_pos:,:]
    neg_sam = targets_pred[:-pred_pos,:]
    
    
    insert_arr = np.array((np.floor(np.linspace(0,len(lbl_),pred_pos))),dtype = int)
    insert_arr[-1] = -1
    base_arr = np.array(neg_sam[:,1],dtype = int)
    for index,item in enumerate(insert_arr):
        base_arr = np.insert(base_arr,item,pos_sam[index,1],0)

    lbl_resample = lbl_[base_arr]  #lbl_trim2 1610 x 5575 的全部数据库
    lbl_resample_target = targets[:,pred]
    
    
    
    lbl_resample_target = np.column_stack([targets, ids, subj, time])
    lbl_resample_target = lbl_resample_target[np.array(lbl_resample[:,-3],dtype = int)]
    
    lbl_resample_target =  lbl_resample_target.astype(np.int)  
    lbl_resample_target =  lbl_resample_target[:,0:15]
  


    train_time = 10
    
    lbl_biassam = lbl_resample[:,0:5572]    # 插完  
    lbl_biastar = lbl_resample_target       # 插完  
    #seperate=int(len(lbl_biassam)/train_time)
    
    #lbl_train_whole = lbl_biassam[0:(1610-seperate),:] 
    #lbl_train_target_whole = lbl_biastar[0:(1610-seperate),:] 
    
    
    #lbl_test_whole = lbl_biassam[(1610-seperate):,:]  
    
    #lbl_test_target_whole = lbl_biastar[(1610-seperate):,:]
    
    
    
    fold = 10
    
    foldidx = int(len(lbl_resample)/fold)
    
    
    
    lbl_train = list()
    lbl_train_target = list()
    lbl_test = list()
    lbl_test_target = list()
    
    

    for i in range(0,fold):
        lbl_test.append(lbl_biassam[i*foldidx:foldidx*(i+1)][:])
        lbl_train.append(np.concatenate((lbl_biassam[:i*foldidx][:],lbl_biassam[foldidx*(i+1):][:])))
        lbl_test_target.append(lbl_biastar[i*foldidx:foldidx*(i+1)][:])
        lbl_train_target.append(np.concatenate((lbl_biastar[:i*foldidx][:],lbl_biastar[foldidx*(i+1):][:])))




#############################################  Tensorflow ############################################
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import initializers
from keras.utils import to_categorical
import math

BANTCH_SIZE = 64
BATCH_NUM = int(math.ceil(len(lbl_train[0]) / BANTCH_SIZE))
STEP_NUM = 5572       # word length
EMBEDDING_DIM = 50     # word embedding will be 2 dimension for 2d visualization
CLASS_NUM = 1
UNITS_NUM = 100
PROJECTION_NUM = 64
OUTPUT_KEEP_PROB = 0.5
LEARNING_RATE = 0.001
EPOCHS_NUM = 10


# making placeholders for X_train and Y_train
# 声明训练数据的变量以及softmax矩阵的变量，其中input_data和target的维度就是train_data和train_label的第一维度其实是小于等于batch_size的（因为sample_num ％ batch_size不一定等于0)，所以写成None
input_data = tf.placeholder(tf.int64, [None, STEP_NUM], name="input_data")
target = tf.placeholder(tf.int64, [None,CLASS_NUM], name="target_label") 
#length = tf.placeholder(tf.float32, [None], name='rnn_length')

target.shape
# cast our label to float32
target = tf.cast(target,tf.float32)

 
# embedding layer  
embedding = tf.get_variable(name = 'embedding', shape=embed.shape, initializer=tf.constant_initializer(embed), trainable=False)  ## trainable=True
input_embedding = tf.nn.embedding_lookup(embedding, input_data)   #得到与输入 input_data 对应的词向量,最后 input_embedding的维度为[BANTCH_SIZE, STEP_NUM, EMBEDDING_DIM]   None X 5572 X 50  通过input_embedding.shape来查看
#input_embedding = tf.unstack(input_embedding_, STEP_NUM, 1)




# 建立lstm节点
# build lstm network


lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(
        num_units=UNITS_NUM,
        #forget_bias=0.3,
        initializer=initializers.xavier_initializer(),
        use_peepholes=False,
        num_proj=PROJECTION_NUM,
        name='LSTM_CELL_1')

#lstm_cell_2 = tf.contrib.rnn.LSTMCell(
#        num_units=UNITS_NUM,
        #forget_bias=0.3,
#        use_peepholes=False,
#        initializer=initializers.xavier_initializer(),
#        num_proj=PROJECTION_NUM,
#        name='LSTM_CELL_2')




#stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1,lstm_cell_2])
#stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1])



# Extrqact the bacth size - this allows for variable batch size
current_batch_size = tf.shape(input_data)[0]

# Create LSTM Initial State of Zeros 后续的学习中一点一点更新 State
initial_state = lstm_cell_1.zero_state(current_batch_size, tf.float32)

# Wrap our lstm cell in a dropout Wrapper
#stacked_lstm = tf.contrib.rnn.DropoutWrapper(cell = stacked_lstm,
#                                         output_keep_prob = OUTPUT_KEEP_PROB)



# 根据之前建立的lstm节点建立rnn网络，返回值为outputs和state 
# outputs维度通过outputs.shape来查看后为 [BANTCH_SIZE, STEP_NUM , UNITS_NUM], 是最后一层每个 STEP 的输出   None X 5572 X 128
# outputs中包含5572是因为包含了5572个timestep所有的输出。因此后续只需要提取最后一个时刻的即可
# final_state是每行最后一个tensor的集合，维度为[BANTCH_SIZE, UNITS_NUM]
outputs, final_state = tf.nn.dynamic_rnn(cell = lstm_cell_1, 
                                        inputs = input_embedding, 
                                        initial_state = initial_state,
                                        # sequence_length=length,
                                        dtype = tf.float32)




#outputs, final_state = rnn.static_rnn(lstm_cell_1,input_embedding,dtype=tf.float32)  # (28 X None X 28) X (28 X 128) -> 28 X None X 128
# instantiate weights and biases
# softmax_w和softmax_b，就是接在hidden_layer之后的全连接层；它下接hidden_layer，上接pred_class
softmax_w = tf.Variable(tf.truncated_normal( [PROJECTION_NUM, CLASS_NUM],mean=0.0,stddev=0.001), dtype=tf.float32)  # 生成 mean=0，std=1随机值   维度：128 X 1

softmax_b = tf.Variable(tf.constant(0.0, shape=[CLASS_NUM]), dtype=tf.float32)   # 维度：2


# 将最后一个时刻的状态（其实就是叠加到最后的一个STEP_NUM）做softmax得到预测结果。由于上一步骤的outputs维度为 [BANTCH_SIZE, STEP_NUM , UNITS_NUM]，所以需要先做一个矩阵变换, 即 None X 5572 X 128 --> 5572 X None X 128
outputs = tf.transpose(outputs,[1,0,2])  # 5572 X None X 128

prediction = tf.matmul(outputs[-1], softmax_w) + softmax_b  # outputs[-1] 为最后一个时刻的输出: [None,128]. 因此 results = None X 128 X 128 X 2 + 2 = None X 2 + 2 = None X 2
prediction_prob = tf.nn.sigmoid (prediction)


# 逐个元素进行判断，如果相等就是True，不相等，就是False
# predictions - [1,1,0,0]
# labels - [1,0,0,1]
# correct_prediction = [1,0,1,0]
correct_prediction = tf.equal(tf.to_int64(prediction_prob > 0.5),tf.argmax(target,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#accuracy = tf.equal(tf.argmax(tf.nn.sigmoid(prediction),axis=1),tf.argmax(target,axis=1))


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

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction,labels = target))   #交叉熵损失函数
#loss = tf.reduce_mean(tf.nn.softmax(logits = prediction))   #交叉熵损失函数
#loss= tf.keras.losses.categorical_crossentropy( tf.nn.softmax(target),  tf.nn.softmax(prediction), from_logits=False,label_smoothing=0)

# Declare our optimizer, in this case RMS Prop
#optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

#optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
saver = tf.train.Saver()

x_batch = list()
y_batch = list()
x_batch_test = list()
y_batch_test = list()
with tf.Session() as sesh:
   init =tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
   sesh.run(init)
   

   j = 3
   for k in range(1):   
       print("--------------------------------Epoch : ",k,"---------------------------------------")    
       for i in range(BATCH_NUM):  # 0, 1, 2, .... 22
           if i <= BATCH_NUM-1:
               x_batch = lbl_train[j][i * BANTCH_SIZE : i *BANTCH_SIZE + BANTCH_SIZE]
               y_batch = lbl_train_target[j][i * BANTCH_SIZE : i *BANTCH_SIZE + BANTCH_SIZE][:,DISEASE_ID]  # y_batch只存储当前病症的label，所以不是64X15，而是(64,)
               y_batch = np.reshape(y_batch, (-1, 1))
              # y_batch=to_categorical(y_batch,num_classes=2,dtype='int64')
           else:
               x_batch = lbl_train[j][i * BANTCH_SIZE :]
               y_batch = lbl_train_target[j][i * BANTCH_SIZE :][:,DISEASE_ID]
               y_batch = np.reshape(y_batch, (-1, 1))
               #y_batch=to_categorical(y_batch,num_classes=2,dtype='int64')
    
           _, l, a = sesh.run([optimizer, loss, accuracy], feed_dict={ input_data:x_batch, target:y_batch})
           
           if i>0:
               #print("STEP",i,"of",BATCH_NUM, "Loss:", l, "ACC:", a, "AUC:", u)
               print("STEP",i,"of",BATCH_NUM, "Loss:", l)
               predictions = prediction_prob.eval(feed_dict = {input_data:x_batch}) 
               outputs_print = outputs[-1].eval(feed_dict = {input_data:x_batch}) 
               softmax_w_print = softmax_w.eval() 
               softmax_b_print = softmax_w.eval() 
               embedding_print = embedding.eval() 
               
       
   x_batch_test = lbl_test[j]
   y_batch_test = lbl_test_target[j][:,DISEASE_ID]
   y_batch_test = np.reshape(y_batch_test, (-1, 1))
  # y_batch_test=to_categorical(y_batch_test,num_classes=2,dtype='int64')
   print("Testing Accuracy:", sesh.run(accuracy, feed_dict={input_data: x_batch_test, target: y_batch_test}))
       




   # save_path = saver.save(sesh,'/Users/han/Desktop/deep learning/model/model.ckpt')
    
    
    
    






