import argparse
import time
import os
import random
import collections
import numpy as np
import torch

from model import DAE, VAE, AAE
from vocab import Vocab
from meter import AverageMeter
from utils import set_seed, logging, load_sent
from batchify import get_batches, generate_sentences


parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument(
    "--train",
    metavar="FILE",
    required=True,
    help="path to training file",
)
parser.add_argument(
    "--valid",
    metavar="FILE",
    required=True,
    help="path to validation file",
)
parser.add_argument(
    "--save-dir",
    default="checkpoints",
    metavar="DIR",
    help="directory to save checkpoints and outputs",
)
parser.add_argument(
    "--load-model",
    default="",
    metavar="FILE",
    help="path to load checkpoint if specified",
)
# Architecture arguments
parser.add_argument(
    "--vocab-size",
    type=int,
    default=10000,
    metavar="N",
    help="keep N most frequent words in vocabulary",
)
parser.add_argument(
    "--dim_z",
    type=int,
    default=128,
    metavar="D",
    help="dimension of latent variable z",
)
parser.add_argument(
    "--dim_emb",
    type=int,
    default=512,
    metavar="D",
    help="dimension of word embedding",
)
parser.add_argument(
    "--dim_h",
    type=int,
    default=1024,
    metavar="D",
    help="dimension of hidden state per layer",
)
parser.add_argument(
    "--nlayers",
    type=int,
    default=1,
    metavar="N",
    help="number of layers",
)
parser.add_argument(
    "--dim_d",
    type=int,
    default=512,
    metavar="D",
    help="dimension of hidden state in AAE discriminator",
)
# Model arguments
parser.add_argument(
    "--model_type",
    default="dae",
    metavar="M",
    choices=["dae", "vae", "aae"],
    help="which model to learn",
)
parser.add_argument(
    "--lambda_kl",
    type=float,
    default=0,
    metavar="R",
    help="weight for kl term in VAE",
)
parser.add_argument(
    "--lambda_adv",
    type=float,
    default=0,
    metavar="R",
    help="weight for adversarial loss in AAE",
)
parser.add_argument(
    "--lambda_p",
    type=float,
    default=0,
    metavar="R",
    help="weight for L1 penalty on posterior log-variance",
)
parser.add_argument(
    "--noise",
    default="0,0,0,0",
    metavar="P,P,P,K",
    help="word drop prob, blank prob, substitute prob" "max word shuffle distance",
)
# Training arguments
parser.add_argument(
    "--dropout",
    type=float,
    default=0.5,
    metavar="DROP",
    help="dropout probability (0 = no dropout)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0005,
    metavar="LR",
    help="learning rate",
)
# parser.add_argument('--clip', type=float, default=0.25, metavar='NORM',
#                    help='gradient clipping')
parser.add_argument(
    "--epochs",
    type=int,
    default=50,
    metavar="N",
    help="number of training epochs",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=256,
    metavar="N",
    help="batch size",
)
# Others
parser.add_argument(
    "--seed",
    type=int,
    default=1111,
    metavar="N",
    help="random seed",
)
parser.add_argument(
    "--no-cuda",
    action="store_true",
    help="disable CUDA",
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="report interval",
)


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        set_seed(self.opt.seed)
        self.setup()

    def setup(self):
        self.setup_logs()
        self.setup_device()
        self.setup_data()
        self.setup_model()

    def setup_logs(self):
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)
        self.log_file = os.path.join(self.opt.save_dir, "log.txt")
        logging(str(self.opt), self.log_file)

    def setup_data(self):
        self.vocab = Vocab()
        logging("# vocab size {}".format(self.vocab.size), self.log_file)

    def setup_device(self):
        cuda = not self.opt.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

    def setup_model(self):
        models = {"dae": DAE, "vae": VAE, "aae": AAE}
        model = models[self.opt.model_type](self.vocab, self.opt).to(self.device)

        if self.opt.load_model:
            ckpt = torch.load(self.opt.load_model)
            model.load_state_dict(ckpt["model"])
            model.flatten()
        logging(
            "# model parameters: {}".format(
                sum(x.data.nelement() for x in model.parameters())
            ),
            self.log_file,
        )

    def evaluate(self, n_samples=64):
        self.model.eval()
        meters = collections.defaultdict(lambda: AverageMeter())
        with torch.no_grad():
            data = generate_sentences(n_samples)
            batches, _ = get_batches(data, self.vocab, self.opt.batch_size, self.device)
            for inputs, targets in batches:
                losses = self.model.autoenc(inputs, targets)
                for k, v in losses.items():
                    meters[k].update(v.item(), inputs.size(1))
        loss = self.model.loss({k: meter.avg for k, meter in meters.items()})
        meters["loss"].update(loss)
        return meters

    def train(self, n_samples=1024):
        best_val_loss = None
        for epoch in range(self.opt.epochs):

            data = generate_sentences(n_samples)
            batches, _ = get_batches(data, self.vocab, self.opt.batch_size, self.device)

            start_time = time.time()
            logging("-" * 80, self.log_file)
            self.model.train()
            meters = collections.defaultdict(lambda: AverageMeter())
            indices = list(range(len(batches)))
            random.shuffle(indices)
            for i, idx in enumerate(indices):
                inputs, targets = batches[idx]
                losses = self.model.autoenc(inputs, targets, is_train=True)
                losses["loss"] = self.model.loss(losses)
                self.model.step(losses)
                for k, v in losses.items():
                    meters[k].update(v.item())

                if (i + 1) % args.log_interval == 0:
                    log_output = "| epoch {:3d} | {:5d}/{:5d} batches |".format(
                        epoch + 1, i + 1, len(indices)
                    )
                    for k, meter in meters.items():
                        log_output += " {} {:.2f},".format(k, meter.avg)
                        meter.clear()
                    logging(log_output, self.log_file)

            valid_meters = self.evaluate()
            logging("-" * 80, self.log_file)
            log_output = "| end of epoch {:3d} | time {:5.0f}s | valid".format(
                epoch + 1, time.time() - start_time
            )
            for k, meter in valid_meters.items():
                log_output += " {} {:.2f},".format(k, meter.avg)
            if not best_val_loss or valid_meters["loss"].avg < best_val_loss:
                log_output += " | saving model"
                ckpt = {"args": args, "model": self.model.state_dict()}
                torch.save(ckpt, os.path.join(args.save_dir, "model.pt"))
                best_val_loss = valid_meters["loss"].avg
            logging(log_output, self.log_file)
        logging("Done training", self.log_file)


if __name__ == "__main__":
    args = parser.parse_args()
    args.noise = [float(x) for x in args.noise.split(",")]
    t = Trainer(args)
    t.train()
