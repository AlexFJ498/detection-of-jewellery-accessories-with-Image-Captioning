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


class CNNModel:
    def __init__(self,
                 model_name=None):
        self.model_name = model_name

        if self.model_name == "inception":
            self.model = InceptionV3(weights='imagenet')
            self.model = Model(inputs=self.model.input,
                               outputs=self.model.layers[-2].output)
            self.width = config.INCEPTION_WIDTH
            self.height = config.INCEPTION_HEIGHT
            self.output_dim = config.INCEPTION_OUTPUT_DIM
            self.preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input

        elif self.model_name == "vgg16":
            self.model = VGG16(weights='imagenet')
            self.model = Model(inputs=self.model.input,
                               outputs=self.model.layers[-2].output)
            self.width = config.VGG16_WIDTH
            self.height = config.VGG16_HEIGHT
            self.output_dim = config.VGG16_OUTPUT_DIM
            self.preprocess_input = tensorflow.keras.applications.vgg16.preprocess_input

        else:
            print("Model name not found")
            sys.exit(1)

    def get_model(self):
        return self.model

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_output_dim(self):
        return self.output_dim

    def get_preprocess_input(self):
        return self.preprocess_input

    def encode_image(self, image):
        image = image.resize((self.width, self.height), Image.ANTIALIAS)

        encoded_image = tensorflow.keras.preprocessing.image.img_to_array(
            image)
        encoded_image = np.expand_dims(encoded_image, axis=0)
        encoded_image = self.preprocess_input(encoded_image)
        encoded_image = self.model.predict(encoded_image)
        encoded_image = np.reshape(encoded_image, self.output_dim)

        return encoded_image


class RNNModel:
    def __init__(self,
                 model_name=None,
                 num_neurons=None,
                 vocab_size=None,
                 max_length=None,
                 output_dim=None,
                 optimizer=None,
                 embedding_matrix=None):

        self.model_name = model_name
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.max_length = max_length
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer

        if model_name == "gru":
            self.loss = config.GRU_LOSS

        else:
            self.loss = config.LSTM_LOSS

    def build_model(self):
        if self.model_name == "lstm":
            self.model = self.build_lstm_model()

        elif self.model_name == "gru":
            self.model = self.build_gru_model()

        else:
            print("Invalid model name")
            sys.exit(1)

    def get_model_name(self):
        return self.model_name

    def get_model(self):
        return self.model

    def get_vocab_size(self):
        return self.vocab_size

    def get_output_dim(self):
        return self.output_dim

    def get_num_neurons(self):
        return self.num_neurons

    def get_max_length(self):
        return self.max_length

    def get_embedding_matrix(self):
        return self.embedding_matrix

    def get_loss(self):
        return self.loss

    def get_optimizer(self):
        return self.optimizer

    def set_model(self, model):
        self.model = model

    def build_lstm_model(self):
        inputs1 = Input(shape=(self.output_dim,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(self.num_neurons, activation='relu')(fe1)
        inputs2 = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size, config.EMBEDDING_SIZE,
                        mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(self.num_neurons)(se2)
        decoder1 = add([fe2, se3])
        decoder2 = Dense(self.num_neurons, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        return model

    def build_gru_model_2(self):
        inputs1 = Input(shape=(self.output_dim,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(self.num_neurons, activation='relu')(fe1)
        inputs2 = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size, config.EMBEDDING_SIZE,
                        mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = GRU(self.num_neurons)(se2)
        decoder1 = add([fe2, se3])
        decoder2 = Dense(self.num_neurons, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        return model

    def build_gru_model(self):
        transfer_values_input = Input(
            shape=(self.output_dim,), name='transfer_values_input')
        decoder_transfer_map = Dense(
            self.num_neurons, activation='tanh', name='decoder_transfer_map')
        decoder_input = Input(shape=(None, ), name='decoder_input')
        decoder_embedding = Embedding(
            input_dim=self.vocab_size, output_dim=300, mask_zero=True, name='decoder_embedding')
        decoder_gru1 = GRU(
            self.num_neurons, name='decoder_gru1', return_sequences=True)
        decoder_gru2 = GRU(
            self.num_neurons, name='decoder_gru2', return_sequences=True)
        decoder_gru3 = GRU(
            self.num_neurons, name='decoder_gru3', return_sequences=True)
        decoder_dense = Dense(
            self.vocab_size, activation='softmax', name='decoder_output')

        initial_state = decoder_transfer_map(transfer_values_input)

        net = decoder_input
        net = decoder_embedding(net)

        net = decoder_gru1(net, initial_state=initial_state)
        net = decoder_gru2(net, initial_state=initial_state)
        net = decoder_gru3(net, initial_state=initial_state)

        decoder_output = decoder_dense(net)
        model = Model(inputs=[transfer_values_input,
                      decoder_input], outputs=[decoder_output])

        return model

    def compile_model(self):
        if self.embedding_matrix is not None:
            self.model.layers[2].set_weights([self.embedding_matrix])
            self.model.layers[2].trainable = True

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss, metrics=config.METRICS)

    def create_generator(self, images, token_captions, batch_size):
        if self.model_name == "lstm":
            return self.create_lstm_generator(images, token_captions, batch_size)

        elif self.model_name == "gru":
            return self.create_gru_generator(images, token_captions, batch_size)

        else:
            print("Invalid model name")
            sys.exit(1)

    def create_lstm_generator(self, images, token_captions, batch_size):
        x1, x2, y = [], [], []
        n = 0

        while True:
            for i in range(len(token_captions)):
                n += 1
                image = images[i]

                seq = token_captions[i]

                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    out_seq = to_categorical(
                        [out_seq], num_classes=self.vocab_size)[0]
                    x1.append(image)
                    x2.append(in_seq)
                    y.append(out_seq)

                if n == batch_size:
                    yield ([np.array(x1), np.array(x2)], np.array(y))

                    x1, x2, y = [], [], []
                    n = 0

    def create_gru_generator(self, images, token_captions, batch_size):
        while True:
            idx = np.random.randint(len(token_captions), size=batch_size)
            transfer_values = images[idx]

            tokens = []
            for i in idx:
                tokens.append(token_captions[i])

            num_tokens = [len(t) for t in tokens]
            max_tokens = np.max(num_tokens)

            tokens_padded = pad_sequences(
                tokens, maxlen=max_tokens, padding='post', truncating='post')

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

    def generate_caption_LSTM(self, image, wordtoidx, idxtoword):
        x1 = []
        x1.append(image)
        in_text = config.START

        for i in range(self.max_length):
            sequence = [wordtoidx[w]
                        for w in in_text.split() if w in wordtoidx]
            sequence = pad_sequences([sequence], maxlen=self.max_length)

            yhat = self.model.predict([np.array(x1), sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = idxtoword[yhat]
            in_text += ' ' + word

            if word == config.STOP:
                break

        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)

        return final

    def generate_caption_GRU(self, photo, wordtoidx, idxtoword, cnn_model, images_path):
        image_path = os.path.join(images_path, photo)
        image = tensorflow.keras.preprocessing.image.load_img(
            image_path, target_size=(cnn_model.height, cnn_model.width))
        image_batch = np.expand_dims(image, axis=0)

        transfer_values = cnn_model.get_model().predict(image_batch)

        shape = (1, self.max_length)
        sequence = np.zeros(shape=shape, dtype=np.int)
        in_text = wordtoidx[config.START]
        output = ""

        for i in range(self.max_length):
            sequence[0, i] = in_text

            x_data = {
                'transfer_values_input': transfer_values,
                'decoder_input': sequence
            }

            yhat = self.model.predict(x_data, verbose=0)
            token_onehot = yhat[0, i, :]

            in_text = np.argmax(token_onehot)

            word = idxtoword[in_text]
            if word == config.STOP:
                break

            if i == 0:
                output += word
            else:
                output += ' ' + word

        return output
