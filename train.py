import sys
import re
import time
from model import *
from utils import *
from os.path import isfile

def load_data():
    data = []
    batch = []
    batch_len = 0 # maximum sequence length of a mini-batch
    print("loading data...")
    word_to_idx = load_vocab(sys.argv[2])
    fo = open(sys.argv[3], "r")
    for line in fo:
        line = line.strip()
        tokens = [int(i) for i in line.split(" ")]
        seq_len = len(tokens)
        if len(batch) == 0: # the first line has the maximum sequence length
            batch_len = seq_len
        pad = [PAD_IDX] * (batch_len - seq_len)
        batch.append(tokens + pad)
        if len(batch) == BATCH_SIZE:
            data.append(Var(LongTensor(batch)))
            batch = []
    fo.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, word_to_idx

def train():
    num_epochs = int(sys.argv[4])
    data, word_to_idx = load_data()
    if VERBOSE:
        idx_to_word = [w for w, _ in sorted(word_to_idx.items(), key = lambda x: x[1])]
    model = rnn(len(word_to_idx))
    optim = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print(model)
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        timer = time.time()
        for x in data:
            loss = 0
            model.zero_grad()
            if VERBOSE:
                pred = [[] for _ in range(BATCH_SIZE)]
            for t in range(x.size(1) - 1):
                y = model(x[:, t].unsqueeze(1)) # teacher forcing
                loss += F.nll_loss(y, x[:, t + 1], size_average = False, ignore_index = PAD_IDX)
                if VERBOSE:
                    for i, j in enumerate(y.data.topk(1)[1]):
                        pred[i].append(scalar(Var(j)))
            loss /= x.data.gt(0).sum() # divide by the number of unpadded tokens
            loss.backward()
            optim.step()
            loss = scalar(loss)
            loss_sum += loss
        timer = time.time() - timer
        loss_sum /= len(data)
        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", "", ei, loss_sum, timer)
        else:
            if VERBOSE:
                for y in pred:
                    print(" ".join([idx_to_word[i] for i in y if i != PAD_IDX]))
            save_checkpoint(filename, model, ei, loss_sum, timer)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model vocab training_data num_epoch" % sys.argv[0])
    print("cuda: %s" % CUDA)
    train()
