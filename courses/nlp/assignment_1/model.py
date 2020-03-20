import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime


class FNNModel(nn.Module):
    def __init__(self,  ntoken,ngram, ninp, nhid, dropout=0.5, tie_weights=False):
        super(FNNModel,self).__init__()
        self.ngram = ngram
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.fc1 = nn.Linear(ninp * ngram, nhid)
        self.tanh = nn.Tanh()
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        # self.softmax = nn.Softmax(dim=1)
        self.init_weights()
        self.time_encoder = 0
        self.time_hidden = 0
        self.time_decoder = 0
        self.time_softmax = 0

    def forward(self, input,):
        time = datetime.now()
        output = self.drop(self.encoder(input))
        output = output.view(output.size(0),-1)
        self.time_encoder += (datetime.now() - time).microseconds

        time = datetime.now()
        output = self.fc1(output)
        output = self.tanh(output)
        self.time_hidden += (datetime.now() - time).microseconds

        time = datetime.now()
        output = self.decoder(output)
        self.time_decoder += (datetime.now() - time).microseconds

        # time = datetime.now()
        # output = self.softmax(output)
        # self.time_softmax += (datetime.now() - time).microseconds
        return output

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
