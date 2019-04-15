from torchnlp.word_to_vector import GloVe
from torchnlp.datasets import snli_dataset
import numpy as np
import torch
import nltk

# Parameters
dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64

# GloVe
print("Loading GloVe... ", end='')
glove = GloVe('840B')
print("Done")


def get_batches(data):
    """
    Generator function that yields shuffled mini-batches. Each batch includes
    a list of tokenized premises, a list of tokenized hypotheses, and a list
    of labels.
    """
    n_samples = len(data)
    remaining_samples = batch_size - (n_samples % batch_size)
    all_idxs = np.append(np.random.permutation(n_samples),
                         np.random.randint(0, n_samples, size=remaining_samples))
    n_batches = len(all_idxs) / batch_size
    batch_idxs = np.split(all_idxs, n_batches)
    for idxs in batch_idxs:
        idxs_list = idxs.tolist()
        p_batch = [nltk.word_tokenize(data[i]['premise']) for i in idxs_list]
        h_batch = [nltk.word_tokenize(data[i]['hypothesis']) for i in idxs_list]
        l_batch = [nltk.word_tokenize(data[i]['label']) for i in idxs_list]
        yield p_batch, h_batch, l_batch


def get_average_embedding_batches(data):
    """
    Get sentence embeddings using the average method in batches.
    """
    for p_batch, h_batch, l_batch in get_batches(data):
        u_batch = []
        v_batch = []
        # for each sentence in a batch
        # get the average of their embeddings
        # append average to u_batch
        for i in range(batch_size):
            u_batch.append(get_sentence_average_embedding(p_batch[i]))
            v_batch.append(get_sentence_average_embedding(h_batch[i]))
        yield u_batch, v_batch, l_batch


def get_sentence_average_embedding(sentence):
    """
    Given a tokenized sentence, it returns the average embedding
    for that sentence
    """
    n_tokens = len(sentence)
    sum = torch.zeros_like(glove[sentence[0]])
    for token in sentence:
        sum += glove[token]
    return sum / n_tokens


if __name__ == '__main__':
    print("This file contains only helper functions")
