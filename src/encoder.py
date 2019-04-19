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


class BiLSTM(nn.Module):

    def __init__(self, maxpooling=False):
        super(BiLSTM, self).__init__()
        self.maxpooling = maxpooling
        self.input_size = 300
        self.hidden_size = 2048
        self.cell = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            bidirectional=True)

    def forward(self, batch_vecs, batch_lens):
        packed_sequence, sorted_lens, undo_indices = pack_batch(batch_vecs, batch_lens)

        packed_y, (h_n, c_n) = self.cell(packed_sequence)

        y_full, y_lens = pad_batch(packed_y, sorted_lens, undo_indices)

        # get the hidden state for every last word
        if self.maxpooling:
            y = self.get_maxpooled_encoding(y_full, y_lens)
        else:
            y = self.get_sentence_encodings(y_full, y_lens)
        return y

    def get_dimensionality(self):
        return self.hidden_size * 2

    def get_maxpooled_encoding(self, y_full, y_lens):
        y = []
        batch_size = len(y_lens)
        for b in range(batch_size):
            maxpool, _ = torch.max(y_full[0:y_lens[b], b, :], dim=0)
            y.append(maxpool)
        return torch.stack(y)

    def get_sentence_encodings(self, y_full, y_lens):
        # in the third dimension, 0 is now forward and 1 backward
        y_unpacked = y_full.view(max(y_lens), 64, 2, self.hidden_size)

        y = []
        for b in range(64):
            forward = y_unpacked[y_lens[b]-1, b, 0, :]
            backward = y_unpacked[0, b, 1, :]
            y.append(torch.cat((forward, backward)))

        return torch.stack(y)