from torchnlp.word_to_vector import GloVe
from torchnlp.datasets import snli_dataset

import utils

##########
# 1. get premise and hypothesis encoding via average
# 2. concatenate u, v, |u-v|, u*v (element-wise)
# 3. feed resulting vector into a 1200 - 512 - 3 mlp with final softmax
# ?. use mini-batches of 64
##########

def main():
    pass


if __name__ == '__main__':
    main()
