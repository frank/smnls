import torchtext
import torch
import nltk
import sys
import os
import numpy as np

from encoder import Baseline, LSTM, BiLSTM
from classifier import MLPClassifier

# ------------------------------ INITIALIZATION --------------------------------
# Cuda parameters
torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Nltk tokenizer
def tokenizer(sentence):
    return [word.lower() for word in nltk.word_tokenize(sentence)]


# Parameters
batch_size = 64
data_limit = 0
encoder_types = [
    'baseline',
    'lstm',
    'bilstm',
    'maxbilstm'
]

# Data loading
text_field = torchtext.data.Field(sequential=True,
                                  tokenize=tokenizer,
                                  include_lengths=True,
                                  use_vocab=True)
label_field = torchtext.data.Field(sequential=False,
                                   use_vocab=True,
                                   pad_token=None,
                                   unk_token=None)
print()
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


def load_model(encoder, classifier, model_path):
    model = torch.load(model_path)
    encoder.load_state_dict(model['encoder_state_dict'])
    classifier.load_state_dict(model['model_state_dict'])
    return encoder, classifier


def test(model_folder='models/', data_path='.data/'):

    train_set, dev_set, test_set = reduce_dataset(full_train_set,
                                                  full_dev_set,
                                                  full_test_set,
                                                  n_samples=data_limit)

    for encoder_type in encoder_types:

        print()
        accuracies = []

        model_path = model_folder + encoder_type + '_model.tar'
        if not os.path.exists(model_path):
            print(encoder_type.upper(), "not found")
            continue
        print("Retrieving", encoder_type.upper(), "model")

        if encoder_type == 'baseline':
            encoder = Baseline().to(device)
        elif encoder_type == 'lstm':
            encoder = LSTM().to(device)
        elif encoder_type == 'bilstm':
            encoder = BiLSTM().to(device)
        elif encoder_type == 'maxbilstm':
            encoder = BiLSTM(maxpooling=True).to(device)
        else:
            encoder = Baseline().to(device)

        classifier = MLPClassifier(encoder, batch_size).to(device)

        encoder, classifier = load_model(encoder,
                                         classifier,
                                         model_path)

        train_iter, dev_iter, test_iter = torchtext.data.BucketIterator.splits(datasets=(train_set, dev_set, test_set),
                                                                               batch_sizes=(
                                                                                   batch_size, batch_size, batch_size),
                                                                               device=device,
                                                                               shuffle=True)

        # iteration of test
        for batch in test_iter:
            # p_batch and h_batch are tuples. The first element is the
            # embedded batch, and the second contains all sentence lengths
            p_batch, h_batch, l_batch = preprocess_batch(batch)

            # forward pass
            preds = classifier.forward(p_batch, h_batch)

            # compute accuracies
            accuracies.append(get_accuracy(preds, l_batch))
        accuracy = np.mean(accuracies)

        print("Accuracy: ", round(accuracy * 100, 1), "%")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_folder = sys.argv[1]
    else:
        model_folder = 'models/'
    if len(sys.argv) > 2:
        data_path = sys.argv[4]
    else:
        data_path = '.data/'
    test(model_folder=model_folder, data_path=data_path)
