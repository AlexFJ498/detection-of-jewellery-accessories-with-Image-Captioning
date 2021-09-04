import os
import numpy as np
import config


def getData(filename):
    data = dict()
    max_length = 0

    with open(filename, "r") as f:
        for line in f.read().split('\n'):
            line = line.split()
            image = line[0]
            caption = line[1:]

            max_length = max(max_length, len(caption))

            if image not in data:
                data[image] = ' '.join(caption)

    max_length += 2

    return data, max_length


def getLexicon(data):
    lex = set()

    for key in data:
        [lex.update(d.split()) for d in data[key].split()]

    lex.update(config.START.split())
    lex.update(config.STOP.split())

    return lex


def getDataArrays(data, train_list):
    images_array = []
    captions_array = []

    with open(train_list, "r") as f:
        for image in f.read().split('\n'):
            images_array.append(image)
            # We include start and stop tokens in each caption
            captions_array.append(
                f'{config.START} {data[image]} {config.STOP}')

    return images_array, captions_array


def getTokenizers(lex):
    idxtoword = {}
    wordtoidx = {}

    idx = 1
    for word in lex:
        wordtoidx[word] = idx
        idxtoword[idx] = word
        idx += 1

    return idxtoword, wordtoidx


def getTokensArrays(captions_array, wordtoidx):
    token_captions_array = []

    for caption in captions_array:
        tokens = []
        words = caption.split()

        for word in words:
            tokens.append(wordtoidx[word])

        token_captions_array.append(tokens)

    return token_captions_array


def getEmbeddings():
    embbedings = {}

    embeddings_path = os.path.join("data", config.EMBEDDING_NAME)
    with open(embeddings_path, 'rb') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embbedings[word] = coefs

        print(f"Found {len(embbedings)} word vectors")

    return embbedings


def getEmbeddingMatrix(embeddings, vocab_size, wordtoidx):
    embedding_matrix = np.zeros((vocab_size, config.EMBEDDING_SIZE))

    for word, i in wordtoidx.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
