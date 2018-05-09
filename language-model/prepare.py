import sys
from model import PAD, EOS, PAD_IDX, EOS_IDX
from utils import tokenize

MIN_LENGTH = 3
MAX_LENGTH = 50

def load_data():
    data = []
    vocab = {PAD: PAD_IDX, EOS: EOS_IDX}
    fo = open(sys.argv[1])
    for line in fo:
        tokens = tokenize(line, "word")
        if len(tokens) < MIN_LENGTH or len(tokens) > MAX_LENGTH:
            continue
        seq = []
        for word in tokens:
            if word not in vocab:
                vocab[word] = len(vocab)
            seq.append(str(vocab[word]))
        data.append(seq)
    data.sort(key = lambda x: len(x), reverse = True)
    fo.close()
    return data, vocab

def save_data(data):
    fo = open(sys.argv[1] + ".csv", "w")
    for seq in data:
        fo.write("%s\n" % " ".join(seq))
    fo.close()

def save_vocab(vocab):
    fo = open(sys.argv[1] + ".vocab", "w")
    for word, _ in sorted(vocab.items(), key = lambda x: x[1]):
        fo.write("%s\n" % word)
    fo.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, vocab = load_data()
    save_data(data)
    save_vocab(vocab)
