import sys
import re
from model import *
from utils import *

def load_model():
    word_to_idx = load_vocab(sys.argv[2])
    idx_to_word = [word for word, _ in sorted(word_to_idx.items(), key = lambda x: x[1])]
    model = rnn(len(word_to_idx))
    print(model)
    load_checkpoint(sys.argv[1], model)
    return model, word_to_idx, idx_to_word

def run_model(model, idx_to_word, data):
    batch = []
    z = len(data)
    eos = [0 for _ in range(z)] # number of EOS tokens in the batch
    while len(data) < BATCH_SIZE:
        data.append(["", [SOS_IDX], []])
    data.sort(key = lambda x: len(x[1]), reverse = True)
    batch_len = len(data[0][1])
    batch = [x[1] + [PAD_IDX] * (batch_len - len(x[1])) for x in data]
    batch = Var(LongTensor(batch))
    lens = batch.data.gt(0).sum(1).tolist()
    dec_out = model(batch, lens)[:, -1]
    dec_in = LongTensor([x[1][-1] for x in data]).unsqueeze(-1)
    t = 0
    while t < 10:
    # while sum(eos) < z:
        if t > 0:
            for i in range(z):
                if eos[i]:
                    continue
                y = int(dec_in[i])
                data[i][2].append(idx_to_word[y])
                if y == EOS_IDX:
                    eos[i] = 1
        dec_out = model(Var(dec_in))
        dec_in = dec_out[:, SOS_IDX + 1:].data.topk(1)[1] + SOS_IDX + 1
        t += 1
    return data[:z]

def predict():
    data = []
    model, word_to_idx, idx_to_word = load_model()
    fo = open(sys.argv[3])
    for line in fo:
        line = line.strip()
        tokens = tokenize(line, "word")
        x = [SOS_IDX] + [word_to_idx[i] for i in tokens]
        data.append([line, x, []])
        if len(data) == BATCH_SIZE:
            result = run_model(model, idx_to_word, data)
            for x in result:
                print(x)
            data = []
    fo.close()
    if len(data):
        result = run_model(model, idx_to_word, data)
        for x in result:
            print(x)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: %s model vocab seed" % sys.argv[0])
    print("cuda: %s" % CUDA)
    predict()
