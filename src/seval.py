from datetime import datetime
import numpy as np
import torchtext
import pickle
import torch
import sys
import os

import SentEval.senteval as senteval

from encoder import Baseline, LSTM, BiLSTM

# ------------------------------ INITIALIZATION --------------------------------
# Cuda parameters
torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
task_size = 'reduced'
encoder_types = [
    'baseline',
    'lstm',
    'bilstm',
    'maxbilstm'
]

print()
print("GloVe:\t\tloading...", end='\r')
glove = torchtext.vocab.GloVe()
print("GloVe:\t\tloaded    ")


# ------------------------------ INITIALIZATION --------------------------------


def load_encoder(encoder, model_path):
    model = torch.load(model_path)
    encoder.load_state_dict(model['encoder_state_dict'])
    return encoder


def create_dictionary(sentences):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    # inverse sort
    sorted_words = sorted(words.items(), key=lambda x: -x[1])
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id


def prepare(params, samples):
    # produce word -> id and inverse mapping
    params.id2word, params.word2id = create_dictionary(samples)

    # set glove as the embedding model
    params.word_vec = glove
    params.wvec_dim = 300


def batcher(params, batch):
    # TODO: make this
    pass


def stest():
    for encoder_type in encoder_types:

        model_path = model_folder + encoder_type + '_model.tar'
        assert os.path.exists(model_path)
        print("Using", encoder_type.upper(), "model")

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

        encoder = load_encoder(encoder, model_path)

        # set parameters for senteval
        params_senteval = {'task_path': 'SentEval/data',
                           'usepytorch': True,
                           'kfold': 5,
                           'classifier': {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                          'tenacity': 3, 'epoch_size': 2},
                           'infersent': encoder.to(device)}
        # senteval engine
        se = senteval.engine.SE(params_senteval, batcher, prepare)

        # task list
        if task_size == 'reduced':
            transfer_tasks = ['MR', 'SICKEntailment', 'STS14', 'STSBenchmark']
        else:
            transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                              'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
                              'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'Length', 'WordContent',
                              'Depth', 'TopConstituents', 'BigramShift', 'Tense', 'SubjNumber',
                              'ObjNumber', 'OddManOut', 'CoordinationInversion']

        # run evaluation
        results = se.eval(transfer_tasks)

        # print
        print(results)

        # define save path
        result_path = 'senteval_results/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = result_path + time_string + '_' + encoder_type

        # save
        with open(file_path, 'wb') as file:
            pickle.dump(results, file)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_folder = sys.argv[1]
    else:
        model_folder = 'models/'
    stest()
