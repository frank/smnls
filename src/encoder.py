import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pack_batch(batch_vecs, batch_lens):
    sorted_batch_lens, indices = batch_lens.sort(0, descending=True)
    sorted_batch_vecs = batch_vecs[:, indices, :]
    undo_indices = torch.tensor([indices[i] for i in indices])
    sorted_batch_lens = sorted_batch_lens.to(torch.int64)
    packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(sorted_batch_vecs, sorted_batch_lens)
    return packed_sequence, sorted_batch_lens, undo_indices


def pad_batch(packed_sequence, sorted_lens, undo_indices):
    y, y_lens = torch.nn.utils.rnn.pad_packed_sequence(packed_sequence)
    y = y[:, undo_indices, :]
    y_lens = sorted_lens[undo_indices]
    return y, y_lens


class Baseline(nn.Module):

    def __init__(self):
        super(Baseline, self).__init__()

    def forward(self, batch_vecs, batch_lens):
        batch_avg = torch.div(torch.sum(batch_vecs, 0),
                              batch_lens.view(-1, 1))
        return batch_avg

    def get_dimensionality(self):
        return 300


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.input_size = 300
        self.hidden_size = 2048
        self.cell = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size)

    def forward(self, batch_vecs, batch_lens):

        print("Input tensor size:", batch_vecs.size())

        packed_sequence, sorted_lens, undo_indices = pack_batch(batch_vecs, batch_lens)
        packed_sequence = packed_sequence.to(device)

        packed_y, (h_n, c_n) = self.cell(packed_sequence)

        y, y_lens = pad_batch(packed_y, sorted_lens, undo_indices)

        print("Output tensor size:", y.size())

        return y

    def get_dimensionality(self):
        return self.hidden_size
