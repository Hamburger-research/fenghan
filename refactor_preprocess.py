# -*- coding: utf-8 -*-
import argparse
import csv
import h5py
import re
import os
import torch
import torch.optim as optim

import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split
import math
from keras.utils.np_utils import to_categorical

from helper import Indexer, load_bin_vec, parse_input_csv, clean_str
from load_data import preprocess, cross_validation, readh5todata

from model import LSTMClassifier
from torch.nn import functional as F

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





def createLossAndOptimizer(model, arg, learning_rate):
    
    #Loss function
    loss = torch.nn.CrossEntropyLoss(weight = weight_scale)
    
    #Optimizer
    if arg.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr= learning_rate)
    elif arg.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), rho = 0.95, lr= learning_rate)
    return(loss, optimizer)
    




def train_model(args, model, learning_rate, batch_size, n_epochs, train_loader):
    import time
    #Print all of the hyperparameters of the training iteration:
    n_batches = len(train_loader)
    loss, optimizer = createLossAndOptimizer(model, args, learning_rate)
    
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("Optimizer= ",args.optimizer)
    print("learning_rate=", learning_rate)
    print("predict_label=", phenotypedictinverse[args.predict_label])
    print("=" * 30)   
    
    
    training_start_time = time.time()
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        #if args.cuda > -1:
            #net.cuda()
        model.train()
        running_loss = 0.0
        print_every = n_batches // 10    # 2
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader,0):  # train_loader size is one fold size divided by 23 bantches. 

            #Get inputs
            inputs, labels = data
                
            #if args.cuda > -1:
                #inputs, labels = inputs.cuda(), labels.cuda() 
            if (inputs.size()[0] is not batch_size):# One of the batch returned by BucketIterator has length different than batch_size 64.
                continue
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = model(inputs)
            print("****************")
            print(labels)
            print("****************")
            print("///////////////")
            print(outputs)
            print("///////////////")
            loss_size = loss(outputs, labels)
            print(loss_size)
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()
            model.resetzeropadding()
            #model.l2norm(args)
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()  
    
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    
    
def predict(args,net):
    
    net.eval()
    with torch.no_grad(): 
        return_predict = torch.tensor([])
        for i, data in enumerate(test_loader,0):
            inputs, labels = data
            if args.cuda > -1:
                inputs, labels = inputs.cuda(), labels.cuda() 
            output = net(inputs)
            _, predicted = torch.max(output, 1)
            if i == 0:
                return_predict = predicted
                continue
            return_predict = torch.cat((return_predict,predicted))
    return return_predict
    

    


def main():

    os.chdir('./')
    global args, word2vec, batch_size, train_set_idx
    global weight_scale, phenotypedictinverse
    phenotypedict = dict({"Cancer":11,"Heart":4,"Lung":5,"Neuro":10,"Pain":9,"Alcohol":7,"Substance":8,"Obesity":1,"Disorders":6,"Depression":12})
    
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    #parser.add_argument('clean_summaries0209.csv', help="Source Input file", type=str)
    #parser.add_argument('word2vec_50d.txt', help="word2vec file", type=str)
    parser.add_argument('--padding', help="padding around each text", type=int, default=4)
    parser.add_argument('--max_note_len', help="Cut off all notes longer than this (0 = no cutoff).", type=int,default=0)
    parser.add_argument('--filename', help="File name for output file", type=str, default="data.h5")
    parser.add_argument('-predict_label', type=int, default= phenotypedict["Depression"] , help= 'Choose which type of phenotyping to detect') 
    parser.add_argument('-topred', type=str, default= "Depression" , help= 'Choose which type of phenotyping to detect') 
    parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 10]')
    parser.add_argument('-batch_size', type=int, default= 64, help='batch size for training [default: 64]')
    parser.add_argument('-output_size', type=int, default= 2, help='final output dim [default: 2]')
    parser.add_argument('-hidden_size', type=int, default= 256, help='output dim of the cell [default: 256]')
    parser.add_argument('-embedding_length', type=int, default= 50, help='number of embedding dimension [default: 50]')
    parser.add_argument('-learning_rate', type=float, default=0.001, help='initial learning rate [default: 0.5]')   
    parser.add_argument('-vocab_size', type=float, default=48849, help='initial learning rate [default: 0.5]')   
    parser.add_argument('-optimizer', type=str, default='Adam', help='optimizer for the gradient descent: Adadelta, Adam')

    
    #    with open("conditions.dict", 'w') as f:
    #        for i, c in enumerate(conditions):
    #            print (f, i + 1, c)


    args = parser.parse_args()

    phenotypedictinverse = dict({11:"Cancer",4:"Heart",5:"Lung",10:"Neuro",9:"Pain",7:"Alcohol",8:"Substance",1:"Obesity",6:"Disorders",12:"Depression"})
    phenotypedictsamples = dict({"Cancer":161,"Heart":275,"Lung":167,"Neuro":368,"Pain":321,"Alcohol":196,"Substance":155,"Obesity":126,"Disorders":295,"Depression":460})
  
    weight_scale = [ 1/ (1610 - phenotypedictsamples[phenotypedictinverse[args.predict_label]]), 1/phenotypedictsamples[phenotypedictinverse[args.predict_label]]]
    #weight_scale = [ phenotypedictsamples[phenotypedictinverse[args.predict_label]]/1610*10, (1610 - phenotypedictsamples[phenotypedictinverse[args.predict_label]])/1610*10]
    weight_scale = torch.FloatTensor(weight_scale)
    # LOAD THE WORD2VEC FILE
    word2vec, emb_size, v_large = load_bin_vec("word2vec_50d.txt") # word2vec whole dataset（label+unlabeled）   470260
    print ('WORD2VEC POINTS:', v_large)
    
    # first step 
    lbl, targets, ids, subj, time, embed = preprocess(args, emb_size, word2vec)
    
    lbl_train, lbl_train_target, lbl_test, lbl_test_target, phenotypedict = cross_validation(lbl, targets, ids, subj, time, args.topred, phenotypedict, phenotypedictsamples)
    
    fold = 10
        
    # put data of  each fold in to a .h5py file
    for i in range(0,fold):  
         with h5py.File('data_biased_'+args.topred+'_cv{0}_occ'.format(i+1) + '0'+'.h5',"w") as f:
             xtrain = np.array(lbl_train[i], dtype=int)
             xtraintarget = np.array(lbl_train_target[i], dtype=int)
             xtest = np.array(lbl_test[i], dtype=int)
             xtesttarget = np.array(lbl_test_target[i], dtype=int)
             f["w2v"] = np.array(embed)
             f['train'] = xtrain
             f['train_label'] = xtraintarget[:,phenotypedict[args.topred]]
             f['test'] = xtest
             f['test_label'] = xtesttarget[:,phenotypedict[args.topred]]    
    
    for i in range(0,fold):
        #torch.cuda.set_device(args.cuda)
        train,test,y_test,w2v = readh5todata(args,'data_biased_'+ phenotypedictinverse[args.predict_label] + '_cv{0}'.format(i+1) + '_occ' + '0' +'.h5')
        args.w2v = w2v
        train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, sampler=None,shuffle = False)
        test_loader = torch.utils.data.DataLoader(test, batch_size= args.batch_size, sampler= None, shuffle = False)
        LSTM = LSTMClassifier(args)
        print(LSTM)
        train_model(args, LSTM, args.learning_rate, args.batch_size, args.epochs, train_loader)
        
        
    
    
   # print('Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
    # save_path = saver.save(sesh,'/Users/han/Desktop/deep learning/model/model.ckpt')
        

  
                       

if __name__ == "__main__":
    main()