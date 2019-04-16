import torchtext
import torch
import nltk
import sys

from encoder import Baseline
from classifier import InferClassifier

# TODO: batch packing and pad packing

################################ INITIALIZATION ################################
dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tokenizer(sentence):
    return [word.lower() for word in nltk.word_tokenize(sentence)]

# Parameters
batch_size = 64
text_field = torchtext.data.Field(sequential=True,
                                  tokenize=tokenizer,
                                  include_lengths=True,
                                  use_vocab=True)
label_field = torchtext.data.Field(sequential=False,
                                   use_vocab=True,
                                   pad_token=None,
                                   unk_token=None)
print("Loading SNLI... ", end='')
train, dev, test = torchtext.datasets.SNLI.splits(text_field, label_field)
print("Done")
print("Loading GloVe... ", end='')
glove = torchtext.vocab.GloVe()
print("Done")
print("Building vocabulary... ", end='')
text_field.build_vocab(train, dev, test, vectors=glove)
label_field.build_vocab(test)
print("Done")
print("Generating batches... ", end='')
train_iter, dev_iter, test_iter = torchtext.data.BucketIterator.splits(datasets=(train, dev, test),
                                                                       batch_sizes=(batch_size, batch_size, batch_size),
                                                                       repeat=False,
                                                                       shuffle=True)
print("Done")
print("Loading embediings... ", end='')
embedding = torch.nn.Embedding.from_pretrained(text_field.vocab.vectors)
print("Done")
################################ INITIALIZATION ################################


def preprocess_batch(batch):
    p_batch = (embedding(batch.premise[0]), batch.premise[1].to(dtype))
    h_batch = (embedding(batch.hypothesis[0]), batch.hypothesis[1].to(dtype))
    l_batch = batch.label
    return p_batch, h_batch, l_batch


def train(args):
    encoder = Baseline()
    classifier = InferClassifier(encoder, batch_size)
    # one iteration of this loop is an epoch
    for batch in train_iter:
        # p_batch and h_batch are tuples. The first element is the
        # embedded batch, and the second contains all sentence lengths
        p_batch, h_batch, l_batch = preprocess_batch(batch)
        out = classifier.forward(p_batch, h_batch)



if __name__ == '__main__':
    train(sys.argv)
