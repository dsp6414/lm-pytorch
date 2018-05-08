import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

BATCH_SIZE = 64
EMBED_SIZE = 500
HIDDEN_SIZE = 1000
NUM_LAYERS = 2
DROPOUT = 0.5
BIDIRECTIONAL = True
NUM_DIRS = 2 if BIDIRECTIONAL else 1
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
VERBOSE = True
SAVE_EVERY = 10

PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence

PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

class rnn(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.hidden = self.init_hidden("GRU") # LSTM or GRU

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.rnn = nn.GRU( # LSTM or GRU
            input_size = EMBED_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = BIDIRECTIONAL
        )
        self.out = nn.Linear(HIDDEN_SIZE, vocab_size)
        self.softmax = nn.LogSoftmax(-1)

        if CUDA:
            self = self.cuda()

    def init_hidden(self, rnn_type): # initialize hidden states
        h = zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # hidden states
        if rnn_type == "LSTM":
            c = zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # cell states
            return (Var(h), Var(c))
        return Var(h)

    def forward(self, x, lens = None):
        x = self.embed(x)
        if lens: # if x.size(1) > 1:
            x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first = True)
            h, _ = self.rnn(x, self.hidden)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)
        else:
            h, _ = self.rnn(x, self.hidden)
        y = self.out(h).squeeze(1)
        y = self.softmax(y)
        return y

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x

def scalar(x):
    return x.view(-1).data.tolist()[0]
