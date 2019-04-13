from torchnlp.word_to_vector import GloVe
import pickle


def main():
    from torchnlp.datasets import snli_dataset
    import nltk

    train, dev, test = snli_dataset(train=True, dev=True, test=True)
    snli_tokens = set()
    print("Tokenizing train set... ", end='')
    for sentence_pair in train:
        tokens = nltk.word_tokenize(sentence_pair['premise'])
        tokens.extend(nltk.word_tokenize(sentence_pair['hypothesis']))
        for token in tokens:
            snli_tokens.add(token)
    print("Done")

    print("Tokenizing dev set... ", end='')
    for sentence_pair in dev:
        tokens = nltk.word_tokenize(sentence_pair['premise'])
        tokens.extend(nltk.word_tokenize(sentence_pair['hypothesis']))
        for token in tokens:
            snli_tokens.add(token)
    print("Done")

    print("Tokenizing test set... ", end='')
    for sentence_pair in test:
        tokens = nltk.word_tokenize(sentence_pair['premise'])
        tokens.extend(nltk.word_tokenize(sentence_pair['hypothesis']))
        for token in tokens:
            snli_tokens.add(token)
    print("Done")

    print("Reading GloVe... ", end='')
    glove = GloVe('840B')
    print("Done")
    snli_tokens = list(snli_tokens)

    common_tokens = []
    for token in snli_tokens:
        if token in glove:
            common_tokens.append(token)

    with open('common_tokens.pkl', 'wb') as handle:
        pickle.dump(common_tokens, handle)


def stripped_glove():
    glove = GloVe('840B')
    with open('common_tokens.pkl', 'rb') as handle:
        common_tokens = pickle.load(handle)
    glove_dic = {}
    for token in common_tokens:
        glove_dic[token] = glove[token]
    glove = None
    return glove_dic


if __name__ == '__main__':
    main()
