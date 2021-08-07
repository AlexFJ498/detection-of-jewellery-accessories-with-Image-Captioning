import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import config
import numpy as np
import platform
import pickle
from PIL import Image
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.preprocessing.image

from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3

from tensorflow.keras.applications import VGG16
import tensorflow.keras.applications.vgg16

from tensorflow.keras.layers import (LSTM, GRU, Embedding, 
                                     TimeDistributed, Dense, RepeatVector, 
                                     Activation, Flatten, Reshape, concatenate,  
                                     Dropout, BatchNormalization, add)

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

def getDataArrays(data):
    images_array = []
    captions_array = []

    for image, caption in data.items():
        images_array.append(image)

        # We include start and stop tokens in each caption
        captions_array.append(f'{config.START} {caption} {config.STOP}')

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

def getCNNModel(name):
    if name == "inception":
        cnn_model = InceptionV3(weights='imagenet')
        cnn_model = Model(cnn_model.input, cnn_model.layers[-2].output)
        width = 299
        height = 299
        output_dim = 2048
        preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input
    elif name == "vgg16":
        cnn_model = VGG16(weights='imagenet')
        cnn_model = Model(cnn_model.input, cnn_model.layers[-2].output)
        width = 224
        height = 224
        output_dim = 4096
        preprocess_input = tensorflow.keras.applications.vgg16.preprocess_input
    
    return cnn_model, width, height, output_dim, preprocess_input

def encodeImage(encode_model, img, width, height, output_dim, preprocess_input):
  img = img.resize((width, height), Image.ANTIALIAS)

  encoded_image = tensorflow.keras.preprocessing.image.img_to_array(img)
  encoded_image = np.expand_dims(encoded_image, axis=0)
  encoded_image = preprocess_input(encoded_image)
  encoded_image = encode_model.predict(encoded_image)
  encoded_image = np.reshape(encoded_image, output_dim )
  
  return encoded_image

def getEmbeddings():
    embbedings = {}

    embeddings_path = os.path.join("data", "SBW-vectors-300-min5.txt")
    with open(embeddings_path, 'rb') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embbedings[word] = coefs
        
        print(f"Found {len(embbedings)} word vectors")
    
    return embbedings

def lstm_generator(images, captions, token_captions_array, max_length, num_photos_per_batch):
  x1, x2, y = [], [], []
  n=0
  while True:
    for i in range(len(captions)):
      n += 1
      image = images[i]

      seq = token_captions_array[i]

      for i in range(1, len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        x1.append(image)
        x2.append(in_seq)
        y.append(out_seq)

      if n == num_photos_per_batch:
        yield ([np.array(x1), np.array(x2)], np.array(y))

        x1, x2, y = [], [], []
        n = 0

def gru_generator(images, token_captions, batch_size):
    while True:
        idx = np.random.randint(len(token_captions), size=batch_size)
        transfer_values = images[idx]

        tokens = []
        for i in idx:
          tokens.append(token_captions[i])

        num_tokens = [len(t) for t in tokens]    
        max_tokens = np.max(num_tokens)

        tokens_padded = pad_sequences(tokens, maxlen=max_tokens, padding='post', truncating='post')
        
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]

        x_data = {
            'decoder_input': decoder_input_data,
            'transfer_values_input': transfer_values
        }

        y_data = {
            'decoder_output': decoder_output_data
        }
        
        yield (x_data, y_data)

def getEmbeddingMatrix(embeddings, vocab_size, wordtoidx):
    embedding_matrix = np.zeros((vocab_size, 300))

    for word, i in wordtoidx.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def buildModelLSTM(vocab_size, output_dim, num_neurons):
    inputs1 = Input(shape=(output_dim,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(num_neurons, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 300, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(num_neurons)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(num_neurons, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    return caption_model

def buildModelGRU(vocab_size, output_dim, num_neurons):
    transfer_values_input = Input(shape=(output_dim,), name='transfer_values_input')
    decoder_transfer_map = Dense(num_neurons, activation='tanh', name='decoder_transfer_map')
    decoder_input = Input(shape=(None, ), name='decoder_input')
    decoder_embedding = Embedding(input_dim=vocab_size, output_dim=300, mask_zero=True, name='decoder_embedding')
    decoder_gru1 = GRU(num_neurons, name='decoder_gru1', return_sequences=True)
    decoder_gru2 = GRU(num_neurons, name='decoder_gru2', return_sequences=True)
    decoder_gru3 = GRU(num_neurons, name='decoder_gru3', return_sequences=True)
    decoder_dense = Dense(vocab_size, activation='softmax', name='decoder_output')

    initial_state = decoder_transfer_map(transfer_values_input)
   
    net = decoder_input
    net = decoder_embedding(net)
    
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    decoder_output = decoder_dense(net)
    caption_model = Model(inputs=[transfer_values_input, decoder_input], outputs=[decoder_output])

    return caption_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to create and train an Image Captioning Model")

    parser.add_argument("--train_path", help="Path to train data", dest="train_path", type=str, required=True)
    parser.add_argument("--model_path", help="Path to save model", dest="model_path", type=str, default="models/")
    parser.add_argument("--cnn", help="Type of Convolutional Neural Network to use (inception or vgg16", dest="cnn_type", type=str, default="inception")
    parser.add_argument("--rnn", help="Type of special units to use for Recurrent Neural Network (lstm or gru)", dest="rnn_type", type=str, default="lstm")
    parser.add_argument("--use_embedding", help="Use or not embedding", dest="use_embedding", type=bool, default=False)
    parser.add_argument("--epochs", help="Number of epochs", dest="epochs", type=int, default=20)
    parser.add_argument("--batch_size", help="Batch size", dest="batch_size", type=int, default=16)
    parser.add_argument("--neurons", help="Number of neurons for each layer of RNN", dest="neurons", type=int, default=256)
    parser.add_argument("--rw", help="Re-write model if already exists.", dest="rewrite", type=bool, default=False)

    args = parser.parse_args()

    # Get path files
    print("Getting data...")
    images_path = os.path.join(args.train_path, "Dataset")
    captions_path = os.path.join(args.train_path, "captions.txt")

    # Get data from files
    data, max_length = getData(captions_path)
    lex = getLexicon(data)

    # Get images and captions arrays
    images_array, captions_array = getDataArrays(data)

    # Tokenize words using idxtoword and wordtoidx
    print("Tokenizing captions...")
    idxtoword, wordtoidx = getTokenizers(lex)
    vocab_size = len(idxtoword) + 1
    
    # Get tokenized captions
    token_captions_array = getTokensArrays(captions_array, wordtoidx)

    # Choose CNN model and get useful params
    print("Creating CNN model...")
    cnn_model, width, height, output_dim, preprocess_input = getCNNModel(args.cnn_type)

    # Encode images and save them
    if platform.system() == "Linux":
        name = args.train_path.split('/')[-1]
    elif platform.system() == "Windows":
        name = args.train_path.split('\\')[-1]

    images_save = os.path.join("data", f'train_images_{name}_{output_dim}.pk1')
    if not os.path.exists(images_save):
        print(f"Encoding images to {images_save}...")
        shape = (len(images_array), output_dim)
        encoded_images = np.zeros(shape=shape, dtype=np.float16)

        for i, img in enumerate(images_array):
            image_path = os.path.join(images_path, img)
            img = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(height, width))
            encoded_images[i] = encodeImage(cnn_model, img, width, height, output_dim, preprocess_input)
        
        with open(images_save, 'wb') as f:
            pickle.dump(encoded_images, f)
            print("Saved encoded images to disk")

    else:
        print(f"Loading images from {images_save}...")
        with open(images_save, 'rb') as f:
            encoded_images = pickle.load(f)

    # Load Embeddings if needed
    if args.use_embedding:
        print("Loading embeddings...")
        embeddings = getEmbeddings()
        embedding_matrix = getEmbeddingMatrix(embeddings, vocab_size, wordtoidx)

    # Build model
    print("Building model...")
    if args.rnn_type == "lstm":
        model = buildModelLSTM(vocab_size, output_dim, args.neurons)
        loss_compile = "categorical_crossentropy"
    elif args.rnn_type == "gru":
        model = buildModelGRU(vocab_size, output_dim, args.neurons)
        loss_compile = "sparse_categorical_crossentropy"

    if args.use_embedding:
        model.layers[2].set_weights([embedding_matrix])
        model.layers[2].trainable = True

    model.compile(loss=loss_compile, optimizer='adam')

    # Train model
    print("Training model...")
    model_save = os.path.join(args.model_path, f'model_{args.cnn_type}_{args.rnn_type}_{args.use_embedding}_{args.epochs}_{args.neurons}.hdf5')
    if not os.path.exists(model_save) or args.rewrite == True:
        if args.rnn_type == "lstm":
            generator = lstm_generator(encoded_images, captions_array, token_captions_array, max_length, args.batch_size)
        elif args.rnn_type == "gru":
            generator = gru_generator(encoded_images, token_captions_array, args.batch_size)

        model.fit(generator, epochs=args.epochs, steps_per_epoch=len(encoded_images) // args.batch_size, verbose=1)

        model.save_weights(model_save)
        print(f'Saved model to {model_save}')
    else:
        model.load_weights(model_save)
        print(f'Model already exists at {model_save}')