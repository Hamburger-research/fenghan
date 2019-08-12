# -*- coding: utf-8 -*-
import argparse
import csv
import h5py
import re
import os

import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split
import math
from keras.utils.np_utils import to_categorical

from helper import Indexer, load_bin_vec, parse_input_csv, clean_str

from model import lstm_model
import tensorflow as tf

# BANTCH_SIZE = 64
# # BATCH_NUM = int(math.ceil(len(lbl_train[0]) / BANTCH_SIZE))
# STEP_NUM = 5572       # word length
# EMBEDDING_DIM = 50     # word embedding will be 2 dimension for 2d visualization
# CLASS_NUM = 1
# UNITS_NUM = 100
# PROJECTION_NUM = 64
# OUTPUT_KEEP_PROB = 0.5
# LEARNING_RATE = 0.001
# EPOCHS_NUM = 10
# args = {}

def preprocess(args, emb_size):

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
    inputs, targets, ids, subj, time = parse_input_csv("./clean_summaries0209.csv", textfield, conditions, id_field, subj_field, chart_field, args)
    
    print("FOUND {} DATA POINTS".format(len(inputs)))


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
    # def split_input_id(data):
        #return data[:, :-3], data[:, -3], data[:, -2], data[:, -1]

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
            #print(key)
            #print (word2vec[key])
        except:
            unks += 1
            pass
    print ("{} UNKNOWN WORDS".format(unks))  

    filename = args.filename


    # STORE ALL THE DATA   
    #tokenizer.write("words.dict")

    return lbl, targets, ids, subj, time, embed


def cross_validation(lbl, targets, ids, subj,time, BANTCH_SIZE, optimizer, loss, accuracy, input_data, target, prediction, all_outputs,
                        softmax_w, softmax_b, embedding, prediction_prob):
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

    x_batch = list()
    y_batch = list()
    x_batch_test = list()
    y_batch_test = list()
    with tf.Session() as sesh:
        init =tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
        sesh.run(init)
           
        
        BATCH_NUM = int(math.ceil(len(lbl_train[0]) / BANTCH_SIZE))
        EPOCHS_NUM = 1

        j = 3
        for k in range(EPOCHS_NUM):  
            
            loss_per_epoch = []
            print("--------BATCH_NUM------------------------Epoch : ",k,"---------------------------------------")    
            for i in range(BATCH_NUM):  # 0, 1, 2, .... 22
                if i <= BATCH_NUM-1:
                     x_batch = lbl_train[j][i * BANTCH_SIZE : i *BANTCH_SIZE + BANTCH_SIZE]
                     y_batch = lbl_train_target[j][i * BANTCH_SIZE : i *BANTCH_SIZE + BANTCH_SIZE][:,DISEASE_ID]  # y_batch只存储当前病症的label，所以不是64X15，而是(64,)
                     y_batch = np.reshape(y_batch, (-1, 1))
                     y_batch=to_categorical(y_batch,num_classes=2,dtype='int64')
                else:
                    x_batch = lbl_train[j][i * BANTCH_SIZE :]
                    y_batch = lbl_train_target[j][i * BANTCH_SIZE :][:,DISEASE_ID]
                    y_batch = np.reshape(y_batch, (-1, 1))
                    y_batch=to_categorical(y_batch,num_classes=2,dtype='int64')
            
                _, l, a = sesh.run([optimizer, loss, accuracy], feed_dict={ input_data:x_batch, target:y_batch})
                   
                if i>0:
                    
                       #print("STEP",i,"of",BATCH_NUM, "Loss:", l, "ACC:", a, "AUC:", u)
                    #print("STEP",i,"of",BATCH_NUM, "Loss:", l)
                    loss_per_epoch.append(np.mean(l))
                    print("STEP",i,"of",BATCH_NUM, "Loss:", np.mean(l))
                    prediction_print = prediction.eval(feed_dict = {input_data:x_batch}) 
                    last_output_print = all_outputs[-1].eval(feed_dict = {input_data:x_batch}) 
                    all_outputs_print = all_outputs.eval(feed_dict = {input_data:x_batch}) 
                    prediction_prob_print = prediction_prob.eval(feed_dict = {input_data:x_batch}) 
                    softmax_w_print = softmax_w.eval() 
                    softmax_b_print = softmax_b.eval() 
                    embedding_print = embedding.eval() 
                    
            print("Epoch ", k, " of ", EPOCHS_NUM, " loss:", np.mean(loss_per_epoch))
                       
               
        x_batch_test = lbl_test[j]
        y_batch_test = lbl_test_target[j][:,DISEASE_ID]
        y_batch_test = np.reshape(y_batch_test, (-1, 1))
        y_batch_test=to_categorical(y_batch_test,num_classes=2,dtype='int64')
        print("Testing Accuracy:", sesh.run(accuracy, feed_dict={input_data: x_batch_test, target: y_batch_test}))
        
    return softmax_w_print, softmax_b_print, embedding_print, all_outputs_print, prediction_print, prediction_prob_print, last_output_print


def main():

    os.chdir('./')
    global args
    global word2vec
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

    #    with open("conditions.dict", 'w') as f:
    #        for i, c in enumerate(conditions):
    #            print (f, i + 1, c)


    args = parser.parse_args()

    # LOAD THE WORD2VEC FILE
    word2vec, emb_size, v_large = load_bin_vec("word2vec_50d.txt") # word2vec 整个数据集（label+unlabeled）   470260
    print ('WORD2VEC POINTS:', v_large)
    
    # first step 
    # X_train, y_train, X_test, y_test = 
    lbl, targets, ids, subj, time, embed = preprocess(args, emb_size)

    # second step
    BANTCH_SIZE, optimizer, loss, accuracy, input_data, target, prediction, all_outputs, softmax_w, softmax_b, embedding ,prediction_prob = lstm_model(embed)

    # third step
    softmax_w_print, softmax_b_print, embedding_print, all_outputs_print, prediction_print, prediction_prob_print, last_output_print = cross_validation(lbl, targets, ids, subj, time, BANTCH_SIZE, optimizer, loss, accuracy, input_data, target, prediction, all_outputs, softmax_w, softmax_b, embedding, prediction_prob)

    
    # save_path = saver.save(sesh,'/Users/han/Desktop/deep learning/model/model.ckpt')
        


                       

if __name__ == "__main__":
    main()