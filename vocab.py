from collections import Counter


class Vocab(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

        specials = ["<pad>", "<go>", "<eos>", "<unk>", "<blank>"]
        for w in specials:
            self.word2idx[w] = len(self.word2idx)
            self.idx2word.append(w)

        for c in range(256):
            w = chr(c)
            self.word2idx[w] = len(self.word2idx)
            self.idx2word.append(w)
        self.size = len(self.word2idx)

        self.pad = self.word2idx["<pad>"]
        self.go = self.word2idx["<go>"]
        self.eos = self.word2idx["<eos>"]
        self.unk = self.word2idx["<unk>"]
        self.blank = self.word2idx["<blank>"]
