import torchtext
import nltk

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
