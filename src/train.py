from torchnlp.word_to_vector import GloVe
from torchnlp.datasets import snli_dataset
import torch
import sys

import utils

# Parameters
dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##########
# 1. get premise and hypothesis encoding via average
# 2. concatenate u, v, |u-v|, u*v (element-wise)
# 3. feed resulting vector into a 1200 - 512 - 3 mlp with final softmax
##########

def main(args):
    """
    Takes args (<model_type> <model_name> <checkpoint_path> <train_data_path>)
    """
    print("Loading SNLI... ", end='')
    test = snli_dataset(test=True)
    print("Done")
    for u_batch, v_batch, l_batch in utils.get_average_embedding_batches(test):
        print(u_batch[0].requires_grad)



if __name__ == '__main__':
    main(sys.argv[1:])
