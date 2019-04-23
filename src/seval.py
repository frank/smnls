from datetime import datetime
import torchtext
import pickle
import torch
import sys
import os

from encoder import Baseline, LSTM, BiLSTM

sys.path.insert(0, 'senteval/')
import senteval

# ------------------------------ INITIALIZATION --------------------------------
# Cuda parameters
torch.set_default_tensor_type(torch.cuda.FloatTensor)
dtype = torch.float
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
    batch = [sent if sent != [] else ['.'] for sent in batch]

    # get useful parameters
    n_sentences = len(batch)
    lengths = [len(tok) for tok in batch]
    max_len = max(lengths)

    # initialize batch elements
    tensor_batch = torch.zeros(max_len, n_sentences, 300)
    batch_lens = torch.tensor(lengths, dtype=dtype)

    # produce embeddings
    for n in range(n_sentences):
        for l, word in enumerate(batch[n]):
            tensor_batch[l, n] = params.word_vec[word]

    batch, batch_lens = (tensor_batch.to(device), batch_lens.to(device))
    embeddings = params['encoder'](batch, batch_lens)

    # embeddings is a np.array containing the sentence embeddings
    return embeddings.cpu().detach().numpy()


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
                           'encoder': encoder.to(device)}
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


def load_results():
    results = {}

    with open('senteval_results/baseline', 'rb') as file:
        results['baseline'] = pickle.load(file)

    with open('senteval_results/lstm', 'rb') as file:
        results['lstm'] = pickle.load(file)

    with open('senteval_results/bilstm', 'rb') as file:
        results['bilstm'] = pickle.load(file)

    with open('senteval_results/maxbilstm', 'rb') as file:
        results['maxbilstm'] = pickle.load(file)

    encoder_types = ['baseline', 'lstm', 'bilstm', 'maxbilstm']

    print("Results on the STS14 multilingual textual similarity task:")

    for encoder_type in encoder_types:
        print("\n############################")
        print(encoder_type.upper(), "encoder:")
        for task in results[encoder_type]:
            print('\n' + task + '-------------\n')
            for measure in results[encoder_type][task]:
                print(measure + ":", results[encoder_type][task][measure])
        print()


if __name__ == '__main__':
    load = False
    if len(sys.argv) > 1:
        if sys.argv[1] == '-l':
            load = True
            model_folder = 'models/'
        else:
            model_folder = sys.argv[1]
    else:
        model_folder = 'models/'
    if not load:
        stest()
    else:
        load_results()

