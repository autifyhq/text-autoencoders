import random

import torch

from vocab import Vocab
from torch.utils.data import Dataset


class RandomSequence(Dataset):
    def __init__(self, max_seq=32):
        self.vocab = Vocab()
        self.max_seq = max_seq

    def __len__(self):
        return 1024 * 1024

    def __getitem__(self, _):
        tokens = []
        n = random.randrange(1, self.max_seq - 1)
        for _ in range(n):
            tokens.append(self.vocab.word2idx[chr(random.randrange(256))])
        padding = [self.vocab.pad] * (self.max_seq - len(tokens) - 1)

        source = [self.vocab.go] + tokens + padding
        target = tokens + [self.vocab.eos] + padding

        return tuple(map(torch.tensor, [source, target]))
