import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Baseline(nn.Module):

    def __init__(self):
        super(Baseline, self).__init__()

    def forward(self, batch_vecs, batch_len):
        batch_avg = torch.div(torch.sum(batch_vecs, 0),
                              batch_len.view(-1, 1))
        return batch_avg


class LSTM(nn.Module):

    def __init(self):
        super(LSTM, self).__init__()
        self.input_size = 300
        self.hidden_size = 2048
        self.cell = nn.LSTMCell(input_size=self.input_size,
                                hidden_size=self.hidden_size)

    def forward(self, batch_vecs, batch_len):
        pass
