# -*- coding: utf-8 -*-
import argparse
import csv
import h5py
import re
import os

import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split


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



def parse_input_csv(filename, textfield, conditions, id_field, subj_field, chart_field, args):
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