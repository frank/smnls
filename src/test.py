import torchtext
import torch
import nltk

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tokenizer(sentence):
    return [word.lower() for word in nltk.word_tokenize(sentence)]

text_field = torchtext.data.Field(sequential=True,
                                  tokenize=tokenizer,
                                  include_lengths=True,
                                  use_vocab=True)

label_field = torchtext.data.Field(sequential=False,
                                   use_vocab=True,
                                   pad_token=None,
                                   unk_token=None)

train, dev, test = torchtext.datasets.SNLI.splits(text_field, label_field)

glove = torchtext.vocab.GloVe()

text_field.build_vocab(train, dev, test, vectors=glove)
label_field.build_vocab(test)

train_iter, dev_iter, test_iter = torchtext.data.BucketIterator.splits(datasets=(train, dev, test),
                                                                       batch_sizes=(5, 5, 5),
                                                                       repeat=False,
                                                                       shuffle=True)

# Vocabulary matrix that can be indexed
embedding = torch.nn.Embedding.from_pretrained(text_field.vocab.vectors)
# Next step: use nn.embeddings

batch = next(iter(train_iter))
batch_premise_embeddings = embedding(batch.premise[0])
# batch packing pad packing torch.nn.sequence something google it up. also pad packing
