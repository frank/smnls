import torchtext
import torch
import nltk
import sys
import numpy as np

from encoder import Baseline, LSTM
from classifier import MLPClassifier

# ------------------------------ INITIALIZATION --------------------------------
torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tokenizer(sentence):
    return [word.lower() for word in nltk.word_tokenize(sentence)]


# Parameters
batch_size = 64
max_epochs = 30
text_field = torchtext.data.Field(sequential=True,
                                  tokenize=tokenizer,
                                  include_lengths=True,
                                  use_vocab=True)
label_field = torchtext.data.Field(sequential=False,
                                   use_vocab=True,
                                   pad_token=None,
                                   unk_token=None)
print("SNLI:\t\tloading...", end='\r')
full_train_set, full_dev_set, full_test_set = torchtext.datasets.SNLI.splits(text_field, label_field)
print("SNLI:\t\tloaded    ")
print("GloVe:\t\tloading...", end='\r')
glove = torchtext.vocab.GloVe()
print("GloVe:\t\tloaded    ")
print("Vocabulary:\tloading...", end='\r')
text_field.build_vocab(full_train_set, full_dev_set, full_test_set, vectors=glove)
label_field.build_vocab(full_test_set)
print("Vocabulary:\tloaded    ")
print("Embeddings:\tloading...", end='\r')
embedding = torch.nn.Embedding.from_pretrained(text_field.vocab.vectors)
embedding.requires_grad = False
print("Embeddings:\tloaded    ")
accuracies = []
# ------------------------------ INITIALIZATION --------------------------------


def reduce_dataset(train_set, dev_set, test_set, n_samples=0):
    if n_samples > 0:
        start = 0
        # train set
        train_end = n_samples if n_samples < len(train_set) else len(train_set)
        train_set.examples = train_set.examples[start:train_end]
        # dev set
        dev_end = n_samples if n_samples < len(dev_set) else len(dev_set)
        dev_set.examples = dev_set.examples[start:dev_end]
        # test set
        test_end = n_samples if n_samples < len(test_set) else len(test_set)
        test_set.examples = test_set.examples[start:test_end]
        print("Reduced sizes:\ttest_set[" + str(train_end) + "]",
              "\n\t\tdev_set[" + str(dev_end) + "]",
              "\n\t\ttest_set[" + str(test_end) + "]\n")
    return train_set, dev_set, test_set


def preprocess_batch(batch):
    p_batch = (embedding(batch.premise[0]), batch.premise[1].to(torch.float))
    h_batch = (embedding(batch.hypothesis[0]), batch.hypothesis[1].to(torch.float))
    l_batch = batch.label.to(torch.long)
    return p_batch, h_batch, l_batch


def get_accuracy(y, t):
    _, y_labels = y.max(1)
    y_labels = y_labels.to(torch.long)
    if len(y) != len(y_labels):
        print("WARNING: size of labels and predictions for last batch don't match")
    accuracies = [1 if y_labels[i] == t[i] else 0 for i in range(len(t))]
    return np.mean(accuracies)


def train(args):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    encoder = LSTM().to(device)
    classifier = MLPClassifier(encoder, batch_size).to(device)

    # optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, weight_decay=0.01)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=0.01)

    cross_entropy = torch.nn.CrossEntropyLoss()

    train_set, dev_set, test_set = reduce_dataset(full_train_set,
                                                  full_dev_set,
                                                  full_test_set,
                                                  n_samples=15)

    # one iteration of this loop is an epoch
    for epoch in range(max_epochs):
        epoch_accuracies = []
        train_iter, dev_iter, test_iter = torchtext.data.BucketIterator.splits(datasets=(train_set, dev_set, test_set),
                                                                               batch_sizes=(
                                                                               batch_size, batch_size, batch_size),
                                                                               device=device,
                                                                               shuffle=True)
        for batch in train_iter:
            # p_batch and h_batch are tuples. The first element is the
            # embedded batch, and the second contains all sentence lengths
            p_batch, h_batch, l_batch = preprocess_batch(batch)

            # forward pass
            preds = classifier.forward(p_batch, h_batch)

            # compute loss
            loss = cross_entropy(preds, l_batch)

            # reset gradients before backwards pass
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

            # compute accuracies
            mb_accuracy = get_accuracy(preds, l_batch)
            epoch_accuracies.append(mb_accuracy)

        print("Epoch", str(epoch + 1) if epoch + 1 > 9 else ' ' + str(epoch + 1),
              "training accuracy:", round(np.mean(epoch_accuracies), 2))


if __name__ == '__main__':
    train(sys.argv)
