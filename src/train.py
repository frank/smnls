import torchtext
from torchtext import data
import random
import torch
import nltk
import sys
import numpy as np

from encoder import Baseline
from classifier import InferClassifier

# TODO: batch packing and pad packing


################################ INITIALIZATION ################################
# dtype = torch.float
torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tokenizer(sentence):
    return [word.lower() for word in nltk.word_tokenize(sentence)]

# Parameters
batch_size = 64
max_epochs = 100
text_field = torchtext.data.Field(sequential=True,
                                  tokenize=tokenizer,
                                  include_lengths=True,
                                  use_vocab=True)
label_field = torchtext.data.Field(sequential=False,
                                   use_vocab=True,
                                   pad_token=None,
                                   unk_token=None)
print("Loading SNLI... ", end='')
full_train_set, full_dev_set, full_test_set = torchtext.datasets.SNLI.splits(text_field, label_field)
print("Done")
print("Loading GloVe... ", end='')
glove = torchtext.vocab.GloVe()
print("Done")
print("Building vocabulary... ", end='')
text_field.build_vocab(full_train_set, full_dev_set, full_test_set, vectors=glove)
label_field.build_vocab(full_test_set)
print("Done")
print("Loading embediings... ", end='')
embedding = torch.nn.Embedding.from_pretrained(text_field.vocab.vectors)
embedding.requires_grad = False
print("Done")
accuracies = []
################################ INITIALIZATION ################################


def reduce_dataset(train_set, dev_set, test_set, percentage=1.):
    if percentage < 1.:
        start = 0
        train_end = int(len(train_set) * percentage)
        train_set.examples = train_set.examples[start:train_end]

        dev_end = int(len(train_set) * percentage)
        dev_set.examples = dev_set.examples[start:dev_end]

        test_end = int(len(train_set) * percentage)
        test_set.examples = test_set.examples[start:test_end]
        print("Reduced test set to", train_end, "samples, dev set to ", dev_end, "samples, test set to ", test_end, "samples")
    return train_set, dev_set, test_set


def preprocess_batch(batch):
    p_batch = (embedding(batch.premise[0]), batch.premise[1].to(torch.float))
    h_batch = (embedding(batch.hypothesis[0]), batch.hypothesis[1].to(torch.float))
    l_batch = batch.label
    return p_batch, h_batch, l_batch


def get_accuracy(y, t):
    _, y_labels = y.max(1)
    y_labels = y_labels.to(torch.float)
    t = t.to(torch.float)
    if len(y) != len(y_labels):
        print("WARNING: size of labels and predictions for last batch don't match")
    accuracies = [1 if y_labels[i] == t[i] else 0 for i in range(len(t))]
    return np.mean(accuracies)


def train(args):
    encoder = Baseline()
    classifier = InferClassifier(encoder, batch_size).to(device=device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, weight_decay=0.99)
    cross_entropy = torch.nn.CrossEntropyLoss()
    train_set, dev_set, test_set = reduce_dataset(full_train_set,
                                                  full_dev_set,
                                                  full_test_set,
                                                  percentage=0.00005)
    # one iteration of this loop is an epoch
    for epoch in range(max_epochs):
        epoch_accuracies = []
        train_iter, dev_iter, test_iter = torchtext.data.BucketIterator.splits(datasets=(train_set, dev_set, test_set),
                                                                               batch_sizes=(batch_size, batch_size, batch_size),
                                                                               device=device,
                                                                               shuffle=True)
        for batch in train_iter:
            # p_batch and h_batch are tuples. The first element is the
            # embedded batch, and the second contains all sentence lengths
            p_batch, h_batch, l_batch = preprocess_batch(batch)
            optimizer.zero_grad()
            preds = classifier.forward(p_batch, h_batch)
            loss = cross_entropy(preds, l_batch)
            loss.backward()
            optimizer.step()
            mb_accuracy = get_accuracy(preds, l_batch)
            epoch_accuracies.append(mb_accuracy)

        print("Epoch", epoch, "accuracy:", round(np.mean(epoch_accuracies), 2))

if __name__ == '__main__':
    train(sys.argv)
