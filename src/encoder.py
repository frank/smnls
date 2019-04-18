import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pack_batch(batch_vecs, batch_lens):
    # ugly fix for the pack_padded_sequence cpu issue
    torch.set_default_tensor_type(torch.FloatTensor)

    # sort the batch lengths in descending order
    sorted_batch_lens, indices = batch_lens.sort(0, descending=True)

    # sort the batch according to the determined order
    sorted_batch_vecs = batch_vecs[:, indices, :]

    # get the indices used later to revert back to the original batch order
    undo_indices = torch.tensor([indices[i] for i in indices])

    # requied for the pack_padded_sequence function
    sorted_batch_lens = sorted_batch_lens.to(torch.int64)
    packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(sorted_batch_vecs, sorted_batch_lens)

    # ugly fix for the pack_padded_sequence cpu issue
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    return packed_sequence, sorted_batch_lens, undo_indices


def pad_batch(packed_sequence, sorted_lens, undo_indices):
    y, y_lens = torch.nn.utils.rnn.pad_packed_sequence(packed_sequence)
    y = y[:, undo_indices, :]
    y_lens = sorted_lens[undo_indices]
    return y, y_lens


class Baseline(nn.Module):

    def __init__(self):
        super(Baseline, self).__init__()
        self.dimensionality = 300

    def forward(self, batch_vecs, batch_lens):
        batch_avg = torch.div(torch.sum(batch_vecs, 0),
                              batch_lens.view(-1, 1))
        return batch_avg

    def get_dimensionality(self):
        return self.dimensionality


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.input_size = 300
        self.hidden_size = 2048
        self.cell = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size)

    def forward(self, batch_vecs, batch_lens):
        packed_sequence, sorted_lens, undo_indices = pack_batch(batch_vecs, batch_lens)

        packed_y, (h_n, c_n) = self.cell(packed_sequence)

        y_full, y_lens = pad_batch(packed_y, sorted_lens, undo_indices)

        # get the hidden state for every last word
        y = torch.stack([y_full[l - 1, i, :] for i, l in enumerate(y_lens)])
        return y

    def get_dimensionality(self):
        return self.hidden_size

