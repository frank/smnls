import sys
import nltk
import torch
import pickle
import torchtext
import numpy as np

from encoder import Baseline, LSTM, BiLSTM
from classifier import MLPClassifier

# Cuda parameters
torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_types = ['baseline', 'lstm', 'bilstm', 'maxbilstm']


# Nltk tokenizer
def tokenizer(sentence):
    return [word.lower() for word in nltk.word_tokenize(sentence)]


def get_sentences(path):
    sentences = []
    with open(path, 'r') as file:
        for line in file:
            sentences.append(line.replace('\n', ''))
    return sentences


def get_batch(sentences):
    pass


def infer():
    """
    Step 1: load premises and hypotheses
    Step 2: convert them into embeddings (keep their txt version saved)
    Step 3:
    """
    premises_txt = get_sentences(premises_path)
    hypotheses_txt = get_sentences(hypotheses_path)

    # both should be tuples. [0] is a torch.tensor of size (max_len x N x 300)
    #                        [1] is a torch.tensor of size (N)



if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        print("Unknown model type. Using BASELINE")
        model_type = 'baseline'
    if len(sys.argv) > 2:
        premises_path = sys.argv[2]
    else:
        premises_path = 'examples/premises.txt'
    if len(sys.argv) > 3:
        hypotheses_path = sys.argv[3]
    else:
        hypotheses_path = 'examples/hypotheses.txt'
    if len(sys.argv) > 4:
        hypotheses_path = sys.argv[4]
    else:
        hypotheses_path = 'examples/hypotheses.txt'
    infer()