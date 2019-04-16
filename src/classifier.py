import torch
import torch.nn as nn

class InferClassifier(nn.Module):

    def __init__(self, encoder, batch_size):
        super(InferClassifier, self).__init__()
        self.encoder = encoder
        self.batch_size = batch_size
        self.input_size = 1200
        self.hidden_size = 512
        self.output_size = 3
        self.classifier = nn.Sequential(
                            nn.Linear(self.input_size, self.hidden_size),
                            nn.Tanh(),
                            nn.Linear(self.hidden_size, self.output_size)
                          )


    def forward(self, p_batch, h_batch):
        p_vecs = p_batch[0]
        p_lens = p_batch[1]

        h_vecs = h_batch[0]
        h_lens = h_batch[1]

        u_batch = self.encoder(p_vecs, p_lens)
        v_batch = self.encoder(h_vecs, h_lens)

        x = self.shared_sentence_encoder(u_batch, v_batch)

        y = self.classifier(x)
        return y


    def shared_sentence_encoder(self, u_batch, v_batch):
        abs_diff = torch.sub(u_batch, v_batch).abs()
        prod = torch.mul(u_batch, v_batch)
        shared_representation = torch.cat((u_batch, v_batch, abs_diff, prod), dim=1)
        return shared_representation
