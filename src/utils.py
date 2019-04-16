import torchtext
import torch
import nltk

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tokenizer(sentence):
    return [word.lower() for word in nltk.word_tokenize(sentence)]

# Parameters
batch_size = 64
# text_field = torchtext.data.Field(sequential=True,
#                                   tokenize=tokenizer,
#                                   include_lengths=True,
#                                   use_vocab=True)
# label_field = torchtext.data.Field(sequential=False,
#                                    use_vocab=True,
#                                    pad_token=None,
#                                    unk_token=None)
# glove = torchtext.vocab.GloVe()


def process_batch(batch, embedding):
    p_batch = (embedding(batch.premise[0]), batch.premise[1])
    h_batch = (embedding(batch.hypothesis[0]), batch.hypothesis[1])
    l_batch = batch.label
    return p_batch, h_batch, l_batch


def get_average_embeddings(p_batch, h_batch):
    """
    Get sentence embeddings using the average method in batches.
    """
    # batch.size() = [max_len, batch_size, embedding_size]
    u_batch = torch.div(torch.sum(p_batch[0], 0),
                        p_batch[1].view(batch_size, 1))
    v_batch = torch.div(torch.sum(h_batch[0], 0),
                        h_batch[1].view(batch_size, 1))
    return u_batch, v_batch


if __name__ == '__main__':
    print("This file contains only helper functions")
