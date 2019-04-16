import torchtext
import torch
import nltk
import sys

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


def train(args):
    for batch in train_iter:
        p_batch = (embedding(batch.premise[0]), batch.premise[1])
        h_batch = (embedding(batch.hypothesis[0]), batch.hypothesis[1])
        l_batch = batch.label
        # batch.size() = [max_len, batch_size, embedding_size]
        u_batch = torch.div(torch.sum(p_batch[0], 0),
                            p_batch[1].view(batch_size, 1))
        v_batch = torch.div(torch.sum(h_batch[0], 0),
                            h_batch[1].view(batch_size, 1))
        print("u_batch:", u_batch[0].size())
        print(u_batch)
        print("v_batch:", v_batch[0].size())
        print(v_batch)
        input()


if __name__ == '__main__':
    train(sys.argv)
