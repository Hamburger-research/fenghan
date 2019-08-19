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
import torch
import torch.utils.data as data_utils




def preprocess(args, emb_size, word2vec):

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
    embed = np.random.uniform(-0.25, 0.25, (len(tokenizer.d) + 1, emb_size))   # 44848 X 50 ; tokenizer 存储了 44848 个 来自 inputs 中的有效单词
    #embed = np.row_stack((np.zeros(50),embed))  #  for padding
    
    
    unks = 0
    
    for key, value in tokenizer.d.items():  #  字典 tokenizer.d:  '<unk>': 1, 'padding': 2, 'Admission': 3, 'Date': 4 ...
        try:
             #-1 because of 1 indexing of word2idx (easier with torch)
            embed[value] = word2vec[key]   # 例如： embed[2] <-- word2vec['Admission'] 
        except:
            unks += 1
            pass
    print ("{} UNKNOWN WORDS".format(unks))    
    
    embed[2,:] = 0

    # STORE ALL THE DATA   
    #tokenizer.write("words.dict")

    
    filename = args.filename


    # STORE ALL THE DATA   
    #tokenizer.write("words.dict")

    return lbl, targets, ids, subj, time, embed






def cross_validation(lbl, targets, ids, subj, time, topred, phenotypedict, phenotypedictsamples):
    
    lbl_ = np.array(lbl, dtype=int)
    targets_ = np.array(targets, dtype=int)

    dataset_concatenate = np.concatenate((lbl_, targets_),axis=1)

    disease_id = phenotypedict[topred]  # 第6种病， 论文中的 Psychiatric disorders


    pred = phenotypedict[topred]  #12

    pred_pos = phenotypedictsamples[topred]

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

    lbl_biassam = lbl_resample[:,4:5572]    # 插完  
    lbl_biastar = lbl_resample_target       # 插完  

    fold = 10

    foldidx = int(len(lbl_resample)/fold)



    lbl_train = list()
    lbl_train_target = list()
    lbl_test = list()
    lbl_test_target = list()



    for i in range(0,fold):
        lbl_test.append(lbl_biassam[i*foldidx:foldidx*(i+1)][:])
        lbl_test_target.append(lbl_biastar[i*foldidx:foldidx*(i+1)][:])
        
        lbl_train.append(np.concatenate((lbl_biassam[:i*foldidx][:],lbl_biassam[foldidx*(i+1):][:])))
        lbl_train_target.append(np.concatenate((lbl_biastar[:i*foldidx][:],lbl_biastar[foldidx*(i+1):][:])))
        


    return lbl_train, lbl_train_target, lbl_test, lbl_test_target, phenotypedict



def readh5todata(args,path):
    myFile = h5py.File(path, 'r')
    x_train = myFile['train'][...]
    seq_lengths_xtrain = []
    for item in x_train:
        current_sample_removedpad = [movepad for movepad in item if movepad!=2]
        seq_lengths_xtrain.append(len(current_sample_removedpad))
    
    y_train = myFile['train_label'][...]
    # y_train = y_train[:,0]
	#seq_lengths.sort(reverse = True) 
    x_train = torch.LongTensor(x_train)
    y_train = torch.LongTensor(y_train)
    seq_lengths_xtrain = torch.LongTensor(seq_lengths_xtrain)
    train = data_utils.TensorDataset(x_train,seq_lengths_xtrain, y_train)
    #y_train = torch.tensor(y_train)
    x_test = myFile['test'][...]
    seq_lengths_xtest = []
    for item in x_test:
        current_sample_removedpad = [movepad for movepad in item if movepad!=2]
        seq_lengths_xtest.append(len(current_sample_removedpad))
    
    y_test = myFile['test_label'][...]
  #  y_test = y_test[:,0]
    x_test = torch.LongTensor(x_test)
    y_test = torch.LongTensor(y_test)
    seq_lengths_xtest = torch.LongTensor(seq_lengths_xtest)
    test = data_utils.TensorDataset(x_test,seq_lengths_xtest, y_test)
    #y_test = y_test
    w2v = myFile['w2v'][...]
    #w2v = np.row_stack((np.zeros(50),w2v))
    w2v[2,:] = 0
    return train,test,y_test,w2v




