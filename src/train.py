import torchtext
import torch
import nltk
import sys
import os
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
# to run tensorboard later, go in the run folder and run
# tensorboard --logdir ./ --host=127.0.0.1

from encoder import Baseline, LSTM, BiLSTM
from classifier import MLPClassifier

# ------------------------------ INITIALIZATION --------------------------------
# Cuda parameters
torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Nltk tokenizer
def tokenizer(sentence):
    return [word.lower() for word in nltk.word_tokenize(sentence)]


# Parameters
lr = 0.1
lr_decay = 0.2
lr_weight_decay = 0.99
weight_decay = 1e-3
lr_threshold = 1e-5
batch_size = 64
max_epochs = 50
data_limit = 0
train_accuracies = []
dev_accuracies = []
train_losses = []
dev_losses = []
model_path = 'models/'
checkpoint_name = 'checkpoint.tar'
encoder_types = [
    'baseline',
    'lstm',
    'bilstm',
    'maxbilstm'
]

# Data loading
text_field = torchtext.data.Field(sequential=True,
                                  tokenize=tokenizer,
                                  include_lengths=True,
                                  use_vocab=True)
label_field = torchtext.data.Field(sequential=False,
                                   use_vocab=True,
                                   pad_token=None,
                                   unk_token=None)
print()
print("SNLI:\t\tloading...", end='\r')
full_train_set, full_dev_set, full_test_set = torchtext.datasets.SNLI.splits(text_field, label_field)
print("SNLI:\t\tloaded    ")
print("GloVe:\t\tloading...", end='\r')
glove = torchtext.vocab.GloVe()
print("GloVe:\t\tloaded    ")
print("Vocabulary:\tloading...", end='\r')
text_field.build_vocab(full_train_set, full_dev_set, full_test_set, vectors=glove)
label_field.build_vocab(full_test_set)
print("Vocabulary:\tloaded    ")
print("Embeddings:\tloading...", end='\r')
embedding = torch.nn.Embedding.from_pretrained(text_field.vocab.vectors)
embedding.requires_grad = False
print("Embeddings:\tloaded    ")
# ------------------------------ INITIALIZATION --------------------------------


def reduce_dataset(train_set, dev_set, test_set, n_samples=0):
    if n_samples > 0:
        start = 0
        # train set
        train_end = n_samples if n_samples < len(train_set) else len(train_set)
        train_set.examples = train_set.examples[start:train_end]
        # dev set
        dev_end = n_samples if n_samples < len(dev_set) else len(dev_set)
        dev_set.examples = dev_set.examples[start:dev_end]
        # test set
        test_end = n_samples if n_samples < len(test_set) else len(test_set)
        test_set.examples = test_set.examples[start:test_end]
        print("Reduced sizes:\ttest_set[" + str(train_end) + "]",
              "\n\t\tdev_set[" + str(dev_end) + "]",
              "\n\t\ttest_set[" + str(test_end) + "]\n")
    return train_set, dev_set, test_set


def preprocess_batch(batch):
    p_batch = (embedding(batch.premise[0]), batch.premise[1].to(torch.float))
    h_batch = (embedding(batch.hypothesis[0]), batch.hypothesis[1].to(torch.float))
    l_batch = batch.label.to(torch.long)
    return p_batch, h_batch, l_batch


def get_accuracy(y, t):
    _, y_labels = y.max(1)
    y_labels = y_labels.to(torch.long)
    if len(y) != len(y_labels):
        print("WARNING: size of labels and predictions for last batch don't match")
    accuracies = [1 if y_labels[i] == t[i] else 0 for i in range(len(t))]
    return np.mean(accuracies)


def save_checkpoint(classifier, optimizer, encoder, epoch, best_dev_accuracy, checkpoint_path='.checkpoint/'):
    global train_accuracies
    global dev_accuracies
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    print("Checkpoint:\tsaving...", end='\r')
    torch.save({"model_state_dict": classifier.state_dict(),
                "encoder_state_dict": encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_accuracies": train_accuracies,
                "dev_accuracies": dev_accuracies,
                "train_losses": train_losses,
                "dev_losses": dev_losses,
                "epoch": epoch,
                "best_dev_accuracy": best_dev_accuracy},
               checkpoint_path + checkpoint_name)
    print("Checkpoint:\tsaved    ")


def load_checkpoint(classifier, optimizer, encoder, epoch, best_dev_accuracy, checkpoint_path='.checkpoint/'):
    global train_accuracies
    global dev_accuracies
    global train_losses
    global dev_losses
    if os.path.exists(checkpoint_path + checkpoint_name):
        print("Checkpoint:\tloading...", end='\r')
        checkpoint = torch.load(checkpoint_path + checkpoint_name)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_accuracies = checkpoint['train_accuracies']
        dev_accuracies = checkpoint['dev_accuracies']
        train_losses = checkpoint['train_losses']
        dev_losses = checkpoint['dev_losses']
        epoch = checkpoint['epoch'] + 1
        best_dev_accuracy = checkpoint['best_dev_accuracy']
        print("Checkpoint:\tloaded    ")
    return classifier, optimizer, encoder, epoch, best_dev_accuracy


def save_model(classifier, optimizer, encoder, epoch, best_dev_accuracy):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print("Model:\t\tsaving...", end='\r')
    torch.save({"model_state_dict": classifier.state_dict(),
                "encoder_state_dict": encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_accuracies": train_accuracies,
                "dev_accuracies": dev_accuracies,
                "train_losses": train_losses,
                "dev_losses": dev_losses,
                "epoch": epoch,
                "best_dev_accuracy": best_dev_accuracy},
               model_path + model_name)
    print("Model:\t\tsaved    ")


def train(encoder_type='baseline', checkpoint_path='.checkpoint/'):

    # initialize tensorboardX SummaryWriter
    time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = 'tensorboard/' + time_string + '_' + encoder_type
    writer = SummaryWriter(log_dir=log_dir)

    if encoder_type not in encoder_types:
        encoder_type = 'baseline'
    global checkpoint_name
    global model_name
    checkpoint_name = encoder_type + '_' + checkpoint_name
    model_name = encoder_type + '_' + model_name

    if encoder_type == 'baseline':
        encoder = Baseline().to(device)
    elif encoder_type == 'lstm':
        encoder = LSTM().to(device)
    elif encoder_type == 'bilstm':
        encoder = BiLSTM().to(device)
    elif encoder_type == 'maxbilstm':
        encoder = BiLSTM(maxpooling=True).to(device)
    else:
        encoder = Baseline().to(device)
    print("Encoder:\t" + encoder_type.upper())

    classifier = MLPClassifier(encoder, batch_size).to(device)
    global lr
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    cross_entropy = torch.nn.CrossEntropyLoss()

    train_set, dev_set, test_set = reduce_dataset(full_train_set,
                                                  full_dev_set,
                                                  full_test_set,
                                                  n_samples=data_limit)

    best_dev_accuracy = 0
    start_epoch = 0

    # load checkpoint if it exists
    classifier, optimizer, encoder, start_epoch, best_dev_accuracy = load_checkpoint(classifier,
                                                                                     optimizer,
                                                                                     encoder,
                                                                                     start_epoch,
                                                                                     best_dev_accuracy,
                                                                                     checkpoint_path)

    if start_epoch > 0:
        print("Resuming from epoch " + str(start_epoch) + "...")

    # one iteration of this loop is an epoch
    for epoch in range(start_epoch, max_epochs):
        # accuracies
        epoch_train_accuracies = []
        epoch_dev_accuracies = []
        epoch_train_losses = []
        epoch_dev_losses = []

        train_iter, dev_iter, test_iter = torchtext.data.BucketIterator.splits(datasets=(train_set, dev_set, test_set),
                                                                               batch_sizes=(
                                                                                   batch_size, batch_size, batch_size),
                                                                               device=device,
                                                                               shuffle=True)

        # iteration of training
        for batch in train_iter:
            # p_batch and h_batch are tuples. The first element is the
            # embedded batch, and the second contains all sentence lengths
            p_batch, h_batch, l_batch = preprocess_batch(batch)

            # forward pass
            preds = classifier.forward(p_batch, h_batch)

            # compute loss
            loss = cross_entropy(preds, l_batch)

            # reset gradients before backwards pass
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

            # compute accuracies and losses
            epoch_train_accuracies.append(get_accuracy(preds, l_batch))
            epoch_train_losses.append(loss.item())
        epoch_train_accuracy = np.mean(epoch_train_accuracies)
        train_accuracies.append(epoch_train_accuracy)
        writer.add_scalar("Training accuracy vs epochs", epoch_train_accuracy, epoch)

        epoch_train_loss = np.mean(epoch_train_losses)
        train_losses.append(epoch_train_loss)
        writer.add_scalar("Training loss vs epochs", epoch_train_loss, epoch)

        # iteration of dev
        for batch in dev_iter:
            # p_batch and h_batch are tuples. The first element is the
            # embedded batch, and the second contains all sentence lengths
            p_batch, h_batch, l_batch = preprocess_batch(batch)

            # forward pass
            preds = classifier.forward(p_batch, h_batch)

            # compute loss
            loss = cross_entropy(preds, l_batch)

            # compute accuracies
            epoch_dev_accuracies.append(get_accuracy(preds, l_batch))
            epoch_dev_losses.append(loss.item())
        epoch_dev_accuracy = np.mean(epoch_dev_accuracies)
        dev_accuracies.append(epoch_dev_accuracy)
        writer.add_scalar("Development accuracy vs epochs", epoch_dev_accuracy, epoch)

        epoch_dev_loss = np.mean(epoch_dev_losses)
        dev_losses.append(epoch_dev_loss)
        writer.add_scalar("Development loss vs epochs", epoch_dev_loss, epoch)

        print("Epoch", (str(epoch + 1) if epoch + 1 > 9 else ' ' + str(epoch + 1)) + ":",
              "\tTRAIN =", round(epoch_train_accuracy * 100, 1), "%\n",
              "\t\tDEV   =", round(epoch_dev_accuracy * 100, 1), "%")

        # learning rate update condition
        if epoch == start_epoch:
            best_dev_accuracy = epoch_dev_accuracy
            save_model(classifier,
                       optimizer,
                       encoder,
                       epoch,
                       best_dev_accuracy)
        else:
            if epoch_dev_accuracy < best_dev_accuracy:
                # update learning rate
                lr *= lr_decay
                print("Learning rate updated: lr =", lr)
                for group in optimizer.param_groups:
                    group['lr'] = lr
            if epoch_dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = epoch_dev_accuracy
                # save the model since the dev accuracy went down
                save_model(classifier,
                           optimizer,
                           encoder,
                           epoch,
                           best_dev_accuracy)

        writer.add_scalar("Learning rate vs epochs", lr, epoch)

        # termination condition
        if lr < lr_threshold:
            print("Training complete")
            break

        save_checkpoint(classifier,
                        optimizer,
                        encoder,
                        epoch,
                        best_dev_accuracy,
                        checkpoint_path)

    if os.path.exists(checkpoint_path + checkpoint_name):
        os.remove(checkpoint_path + checkpoint_name)
    writer.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        encoder_type = sys.argv[1].lower()
    else:
        encoder_type = 'baseline'
    if len(sys.argv) > 2:
        model_name = sys.argv[2].lower() + '.tar'
    else:
        model_name = 'model.tar'
    if len(sys.argv) > 3:
        checkpoint_path = sys.argv[3]
    else:
        checkpoint_path = '.checkpoint/'
    if len(sys.argv) > 4:
        data_path = sys.argv[4]
    else:
        data_path = '.data/'
    train(encoder_type=encoder_type, checkpoint_path=checkpoint_path)
