import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import pickle
import argparse
import platform
import numpy as np
import dataFunctions
import modelFunctions

from tensorflow import keras
import tensorflow.keras.preprocessing.image

def searchCaption(caption, data):
    predicted_image = False
    for img, cap in data.items():
        if caption == cap:
            predicted_image = True
            break

    return predicted_image, img

def obtainCCR(ccr, total):
    ccr = ccr / total
    ccr = ccr * 100
    return ccr

def getNumAccesories(captions_array):
    num_gargantillas = 0
    num_pendientes = 0
    num_anillos = 0
    num_pulseras = 0

    for caption in captions_array:
        word = caption.split()[1]
        if word == 'Gargantilla' or word == 'Colgante':
            num_gargantillas += 1
        elif word == 'Pendientes':
            num_pendientes += 1
        elif word == 'Anillo' or word == "Sortija":
            num_anillos += 1
        elif word == 'Pulsera':
            num_pulseras += 1

    return num_gargantillas, num_pendientes, num_anillos, num_pulseras

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to create and train an Image Captioning Model")

    parser.add_argument("--train_path", help="Path to train data", dest="train_path", type=str, required=True)
    parser.add_argument("--model_path", help="Path to save model", dest="model_path", type=str, default="models/")
    parser.add_argument("--cnn", help="Type of Convolutional Neural Network to use (inception or vgg16)", dest="cnn_type", type=str, default="inception")
    parser.add_argument("--rnn", help="Type of special units to use for Recurrent Neural Network (lstm or gru)", dest="rnn_type", type=str, default="lstm")
    parser.add_argument("--use_embedding", help="Use or not embedding", dest="use_embedding", type=bool, default=False)
    parser.add_argument("--epochs", help="Number of epochs", dest="epochs", type=int, default=20)
    parser.add_argument("--batch_size", help="Batch size", dest="batch_size", type=int, default=10)
    parser.add_argument("--neurons", help="Number of neurons for each layer of RNN", dest="neurons", type=int, default=256)
    parser.add_argument("--rw_images", help="Re-write encoded images if already exists.", dest="rewrite_images", type=bool, default=False)
    parser.add_argument("--rw_model", help="Re-write model if already exists.", dest="rewrite_model", type=bool, default=False)

    args = parser.parse_args()

    # Get path files
    print("Getting data...")
    images_path = os.path.join(args.train_path, "train")
    captions_path = os.path.join(args.train_path, "captions.txt")
    train_list = os.path.join(args.train_path, "train.txt")

    # Get data from files
    data, max_length = dataFunctions.getData(captions_path)
    lex = dataFunctions.getLexicon(data)

    # Get images and captions arrays
    images_array, captions_array = dataFunctions.getDataArrays(data, train_list)

    # Tokenize words using idxtoword and wordtoidx
    print("Tokenizing captions...")
    idxtoword, wordtoidx = dataFunctions.getTokenizers(lex)
    vocab_size = len(idxtoword) + 1
    
    # Get tokenized captions
    token_captions_array = dataFunctions.getTokensArrays(captions_array, wordtoidx)

    # Choose CNN model and get useful params
    print("Creating CNN model...")
    cnn_model, width, height, output_dim, preprocess_input = modelFunctions.getCNNModel(args.cnn_type)

    # Encode train images and save them
    if platform.system() == "Linux":
        name = args.train_path.split('/')[-1]
    elif platform.system() == "Windows":
        name = args.train_path.split('\\')[-1]

    train_images_save = os.path.join("data", f'train_images_{name}_{output_dim}.pk1')
    if not os.path.exists(train_images_save) or args.rewrite_images == True:
        print(f"Encoding images to {train_images_save}...")
        shape = (len(images_array), output_dim)
        train_encoded_images = np.zeros(shape=shape, dtype=np.float16)

        for i, img in enumerate(images_array):
            image_path = os.path.join(images_path, img)
            img = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(height, width))
            train_encoded_images[i] = modelFunctions.encodeImage(cnn_model, img, width, height, output_dim, preprocess_input)
        
        with open(train_images_save, 'wb') as f:
            pickle.dump(train_encoded_images, f)
            print("Saved encoded images to disk")
    else:
        print(f"Loading images from {train_images_save}...")
        with open(train_images_save, 'rb') as f:
            train_encoded_images = pickle.load(f)

    # Load Embeddings if needed
    if args.use_embedding == True:
        print("Loading embeddings...")
        embeddings = dataFunctions.getEmbeddings()
        embedding_matrix = dataFunctions.getEmbeddingMatrix(embeddings, vocab_size, wordtoidx)

    # Build model
    print("Building model...")
    if args.rnn_type == "lstm":
        model = modelFunctions.buildModelLSTM(vocab_size, output_dim, args.neurons, max_length)
        loss_compile = "categorical_crossentropy"
    elif args.rnn_type == "gru":
        model = modelFunctions.buildModelGRU(vocab_size, output_dim, args.neurons)
        loss_compile = "sparse_categorical_crossentropy"
    else:
        print("Invalid RNN type")
        sys.exit(1)

    if args.use_embedding == True:
        model.layers[2].set_weights([embedding_matrix])
        model.layers[2].trainable = True

    model.compile(loss=loss_compile, optimizer='adam')

    # Train model
    print("Training model...")
    model_save = os.path.join(args.model_path, f'model_{args.cnn_type}_{args.rnn_type}_{args.use_embedding}_{args.epochs}_{args.neurons}.hdf5')
    if not os.path.exists(model_save) or args.rewrite_model == True:
        for i in range(args.epochs*2):
            if args.rnn_type == "lstm":
                generator = modelFunctions.lstm_generator(train_encoded_images, captions_array, token_captions_array, max_length, args.batch_size, vocab_size)
            elif args.rnn_type == "gru":
                generator = modelFunctions.gru_generator(train_encoded_images, token_captions_array, args.batch_size)

            model.fit(generator, epochs=1, steps_per_epoch=(len(train_encoded_images) // args.batch_size), verbose=1)

        model.optimizer.lr = 1e-4

        for i in range(args.epochs):
            if args.rnn_type == "lstm":
                generator = modelFunctions.lstm_generator(train_encoded_images, captions_array, token_captions_array, max_length, args.batch_size, vocab_size)
            elif args.rnn_type == "gru":
                generator = modelFunctions.gru_generator(train_encoded_images, token_captions_array, args.batch_size)

            model.fit(generator, epochs=1, steps_per_epoch=(len(train_encoded_images) // args.batch_size), verbose=1)

        model.save(model_save)
        print(f'Saved model to {model_save}')

        with open(os.path.join(args.model_path, f'idxtoword_{args.cnn_type}_{args.rnn_type}_{args.use_embedding}_{args.epochs}_{args.neurons}.pk1'), 'wb') as f:
            pickle.dump(idxtoword, f)
            print("Saved idxtoword to disk")

        with open(os.path.join(args.model_path, f'wordtoidx_{args.cnn_type}_{args.rnn_type}_{args.use_embedding}_{args.epochs}_{args.neurons}.pk1'), 'wb') as f:
            pickle.dump(wordtoidx, f)
            print("Saved wordtoidx to disk")
    else:
        print(f'The model already exists at {model_save}')