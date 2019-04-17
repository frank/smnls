# import torchtext
# import torch
# import nltk
#
# from encoder import Baseline
# from classifier import InferClassifier
#
# batch_size = 64
# dtype = torch.float
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# encoder = Baseline()
# classifier = InferClassifier(encoder, batch_size).to(device=device)

import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.cuda.is_available()
torch.cuda.current_device()
