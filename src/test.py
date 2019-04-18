import torch
import pickle

from encoder import Baseline, LSTM
from classifier import MLPClassifier

# ------------------------------ INITIALIZATION --------------------------------
torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 64

with open("batch_ex.pkl", "rb") as file:
    batch = pickle.load(file)

p_batch, h_batch, l_batch = batch['p'], batch['h'], batch['l']
# ------------------------------ INITIALIZATION --------------------------------

encoder = LSTM().to(device=device)
classifier = MLPClassifier(encoder, batch_size).to(device=device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=0.01)
cross_entropy = torch.nn.CrossEntropyLoss()

for epoch in range(20):
    preds = classifier.forward(p_batch, h_batch)

    # compute loss
    loss = cross_entropy(preds, l_batch)

    # reset gradients before backwards pass
    optimizer.zero_grad()

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()
    print("Epoch", str(epoch + 1) if epoch + 1 > 9 else ' ' + str(epoch + 1),
          "training loss:", round(loss.item(), 3))
