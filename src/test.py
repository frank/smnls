import torchtext
import torch
import nltk

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
