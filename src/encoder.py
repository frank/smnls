import torch
import torch.nn as nn

class Baseline(nn.Module):

    def __init__(self):
        super(Baseline, self).__init__()

    def forward(self, batch_vecs, batch_len):
        batch_avg = torch.div(torch.sum(batch_vecs, 0),
                              batch_len.view(-1, 1))
        return batch_avg
