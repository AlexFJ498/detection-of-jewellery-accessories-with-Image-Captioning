import os
import sys
import numpy as np
import config

from PIL import Image
from tensorflow.keras import Input, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3

from tensorflow.keras.applications import VGG16
import tensorflow.keras.applications.vgg16

from tensorflow.keras.layers import (LSTM, GRU, Embedding, 
                                     TimeDistributed, Dense, RepeatVector, 
                                     Activation, Flatten, Reshape, concatenate,  
                                     Dropout, BatchNormalization, add)

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
    else:
        print("Invalid model name")
        sys.exit(1)
    
    return cnn_model, width, height, output_dim, preprocess_input

def encodeImage(encode_model, img, width, height, output_dim, preprocess_input):
  img = img.resize((width, height), Image.ANTIALIAS)

  encoded_image = tensorflow.keras.preprocessing.image.img_to_array(img)
  encoded_image = np.expand_dims(encoded_image, axis=0)
  encoded_image = preprocess_input(encoded_image)
  encoded_image = encode_model.predict(encoded_image)
  encoded_image = np.reshape(encoded_image, output_dim )
  
  return encoded_image

def lstm_generator(images, captions, token_captions_array, max_length, num_photos_per_batch, vocab_size):
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

def buildModelLSTM(vocab_size, output_dim, num_neurons, max_length):
    inputs1 = Input(shape=(output_dim,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(num_neurons, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, config.EMBEDDING_SIZE, mask_zero=True)(inputs2)
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

def generateCaptionLSTM(image, wordtoidx, idxtoword, model, max_length):
    x1 = []
    x1.append(image)
    in_text = config.START
    for i in range(max_length):
      sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
      sequence = pad_sequences([sequence], maxlen=max_length)

      yhat = model.predict([np.array(x1),sequence], verbose=0)
      yhat = np.argmax(yhat)
      word = idxtoword[yhat]
      in_text += ' ' + word
      if word == config.STOP:
        break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def generateCaptionGRU(photo, wordtoidx, idxtoword, model, cnn_model, height, width, images_path, max_length):
    image_path = os.path.join(images_path, photo)
    image = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(height, width))
    image_batch = np.expand_dims(image, axis=0)

    transfer_values = cnn_model.predict(image_batch)

    shape = (1, max_length)
    sequence = np.zeros(shape=shape, dtype=np.int)
    in_text = wordtoidx[config.START]
    output = ""

    for i in range(max_length):
      sequence[0, i] = in_text

      x_data = {
          'transfer_values_input': transfer_values,
          'decoder_input': sequence
      }

      yhat = model.predict(x_data, verbose=0)
      token_onehot = yhat[0, i, :]
    #   print(token_onehot)

      in_text = np.argmax(token_onehot)
      
      word = idxtoword[in_text]
      if word == config.STOP:
        break
      
      if i == 0:
        output += word
      else:
        output += ' ' + word

    return output