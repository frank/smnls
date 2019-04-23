# Learning Sentence Representations from Natural Language Inference Data

This repository contains all the code necessary to train and evaluate
the four models described in [1].

The models may be trained by using:

```bash
$ python train.py <encoder_type>
```

Where **encoder_type** is one of the following: *baseline*, *lstm*,
*bilstm*, *maxbilstm*. These correspond respectively to the baseline
mean encoder, the unidirectional LSTM encoder, the bidirectional
LSTM encoder using concatenation of the last embeddings produced
by each layer, and the bidirectional LSTM encoder using max-pooling.

All four models can be evaluated on the SNLI dataset by running:

```bash
$ python eval.py
```

The four models can also be evaluated on several transfer tasks using
the SentEval framework [2] by running:

```bash
$ python seval.py
```

The results of the SentEval test are both printed and saved. They can
later be printed again by running:

```bash
$ python seval.py -l
```

Finally, each of the four encoders can be tested on a user provided
examples. This may be done by running:

```bash
$ python infer.py <encoder_type> <premises_path> <hypotheses_path>
```

Here, **premises_path** is the location of a text file containing one
premise per line. Conversely, **hypotheses_path** should be the location
of a text file containing one hypothesis per line, where the premise
at line *n* should correspond to the hypothesis at line *n*. The program
will print the prediction.

## Requirements

All scripts require that the SNLI dataset is stored in:

```
./.data/snli/snli_1.0/
```

Also, that the GloVe embeddings are stored in:

```
./.vector_cache/
```

These will be downloaded automatically if they are not already
in the working directory.

Lastly, in order for the SentEval framework to work, the working directory
should contain the SentEval repository, where the contents of the folder:

```
./SentEval/senteval/
```

have been moved to the SentEval folder, which has been renamed
to *senteval*.

### References
[1] [Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault,
Antoine Bordes - Supervised Learning of Universal Sentence
Representations from Natural Language Inference Data,
EMNLP 2017](https://arxiv.org/abs/1705.02364)

[2] [Alexis Conneau, Douwe Kiela - SentEval: An Evaluation Toolkit
for Universal Sentence Representations,
LREC 2018](https://arxiv.org/abs/1803.05449)