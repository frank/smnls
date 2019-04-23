import os
import sys
import nltk
import torch
import torchtext

from encoder import Baseline, LSTM, BiLSTM
from classifier import MLPClassifier

# Cuda parameters
torch.set_default_tensor_type(torch.cuda.FloatTensor)
dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
model_types = ['baseline', 'lstm', 'bilstm', 'maxbilstm']
outcome = {0: "contradiction",
           1: "entailment",
           2: "neutral"}


# Nltk tokenizer
def tokenizer(sentence):
    return [word.lower() for word in nltk.word_tokenize(sentence)]


def get_sentences(path):
    sentences = []
    with open(path, 'r') as file:
        for line in file:
            sentences.append(line.replace('\n', ''))
    return sentences


def get_batch(sentences, vocab):
    # tokenize the sentences
    tokenized = [tokenizer(sentence) for sentence in sentences]

    # get useful parameters
    n_sentences = len(sentences)
    lengths = [len(tok) for tok in tokenized]
    max_len = max(lengths)

    # initialize batch elements
    batch = torch.zeros(max_len, n_sentences, 300)
    batch_lens = torch.tensor(lengths, dtype=dtype)

    # produce embeddings
    for n in range(n_sentences):
        for l, word in enumerate(tokenized[n]):
            batch[l, n] = vocab[word]

    return batch.to(device), batch_lens.to(device)


def load_model(encoder, classifier, model_path):
    model = torch.load(model_path)
    encoder.load_state_dict(model['encoder_state_dict'])
    classifier.load_state_dict(model['model_state_dict'])
    return encoder, classifier


def infer():
    premises_txt = get_sentences(premises_path)
    hypotheses_txt = get_sentences(hypotheses_path)

    # both should be tuples. [0] is a torch.tensor of size (max_len x N x 300)
    #                        [1] is a torch.tensor of size (N)

    # get glove
    print("GloVe:\t\tloading...", end='\r')
    glove = torchtext.vocab.GloVe()
    print("GloVe:\t\tloaded    ")

    # get the batches in the right, embedded format
    p_batch = get_batch(premises_txt, glove)
    h_batch = get_batch(hypotheses_txt, glove)

    # remove glove from memory
    glove = None

    # check that there's an equal number of pairs
    assert p_batch[1].size() == h_batch[1].size()
    n_sentences = len(p_batch[1])

    # load the right model
    model_path = 'models/' + model_type + '_model.tar'
    assert os.path.exists(model_path)
    print("Using", model_type.upper(), "model")

    if model_type == 'baseline':
        encoder = Baseline().to(device)
    elif model_type == 'lstm':
        encoder = LSTM().to(device)
    elif model_type == 'bilstm':
        encoder = BiLSTM().to(device)
    elif model_type == 'maxbilstm':
        encoder = BiLSTM(maxpooling=True).to(device)
    else:
        encoder = Baseline().to(device)

    classifier = MLPClassifier(encoder, n_sentences).to(device)

    encoder, classifier = load_model(encoder,
                                     classifier,
                                     model_path)

    # get predictions
    y = classifier.forward(p_batch, h_batch)
    predictions = [outcome[l] for l in y.argmax(1).cpu().numpy()]

    # print result
    for i in range(n_sentences):
        print("\nPREMISE:\t", premises_txt[i])
        print("HYPOTHESIS:\t", hypotheses_txt[i])
        print("PREDICTION:\t", predictions[i])


if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        model_type = 'baseline'
    if len(sys.argv) > 2:
        premises_path = sys.argv[2]
    else:
        premises_path = 'examples/premises.txt'
    if len(sys.argv) > 3:
        hypotheses_path = sys.argv[3]
    else:
        hypotheses_path = 'examples/hypotheses.txt'
    if model_type not in model_types:
        print("Unknown model type. Using BASELINE")
        model_type = 'baseline'
    infer()
