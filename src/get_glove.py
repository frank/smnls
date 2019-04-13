import numpy as np

from torchnlp.datasets import snli_dataset
from torchnlp.word_to_vector import GloVe

print("Getting glove...")
glove = GloVe('840B')
print("Gotten.")
print("Getting snli...")
train, dev, test = snli_dataset(train=True, dev=True, test=True)
print("Gotten.")