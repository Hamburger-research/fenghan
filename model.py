# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class LSTMClassifier(nn.Module):
	def __init__(self, args):
		super(LSTMClassifier, self).__init__()
        
		
		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		"""
		self.args = args
		self.batch_size = args.batch_size
		self.output_size = args.output_size
		self.hidden_size = args.hidden_size
		self.vocab_size = args.vocab_size
		self.embedding_length = args.embedding_length
		self.word_embeddings = nn.Embedding(args.vocab_size, args.embedding_length)
		self.word_embeddings.weight.data.copy_(torch.from_numpy(args.w2v))
		self.lstm = nn.LSTM(args.embedding_length, args.hidden_size)
		self.label = nn.Linear(args.hidden_size, args.output_size)
		
	def forward(self, input_sentence, batch_size=None):
	
		""" 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
		final_output.shape = (batch_size, output_size)
		
		"""
		
		''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
		input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
		input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
		if batch_size is None:
			h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)) # Initial hidden state of the LSTM
			c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)) # Initial cell state of the LSTM
		else:
			h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
			c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
		output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        
		final_output = self.label(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
		
		return final_output

    
	def resetzeropadding(self):
            parameters = self.word_embeddings.state_dict()
            parameters['weight'][2] = 0
            self.word_embeddings.load_state_dict(parameters)
        
	def printembweight(self):
            parameters = self.word_embeddings.weight.detach().cpu().numpy()
            print(parameters)
        
	#def l2norm(self,args):
           # wei = self.fc1.state_dict()
            #for j in range(0, wei['weight'].size(0)):
                #normnum = wei['weight'][j].norm()
                #wei['weight'][j].mul(args.l2s).div(1e-7 + normnum)
           # self.fc1.load_state_dict(wei) 
    
    
    

    
    