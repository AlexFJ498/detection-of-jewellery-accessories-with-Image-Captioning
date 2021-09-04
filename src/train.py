import tensorflow.keras.preprocessing.image
from modelFunctions import CNNModel, RNNModel
import dataFunctions
import config
import numpy as np
import platform
import argparse
import pickle
from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def plot_image(data1, data2, title):
    plt.plot(data1)
    plt.plot(data2)

    plt.title(title)
    plt.ylabel(title)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to create and train an Image Captioning Model")

    parser.add_argument("--train_path", help="Path to train data",
                        dest="train_path", type=str, required=True)
    parser.add_argument("--model_path", help="Path to save model",
                        dest="model_path", type=str, default="models/")
    parser.add_argument("--cnn", help="Type of Convolutional Neural Network to use (inception, vgg16 or mobilenet)",
                        dest="cnn_type", type=str, default="inception")
    parser.add_argument("--rnn", help="Type of special units to use for Recurrent Neural Network (lstm or gru)",
                        dest="rnn_type", type=str, default="lstm")
    parser.add_argument("--use_embedding", help="Use or not embedding",
                        dest="use_embedding", type=bool, default=False)
    parser.add_argument("--epochs", help="Number of epochs",
                        dest="epochs", type=int, default=50)
    parser.add_argument("--batch_size", help="Batch size",
                        dest="batch_size", type=int, default=10)
    parser.add_argument("--neurons", help="Number of neurons for each layer of RNN",
                        dest="neurons", type=int, default=256)
    parser.add_argument("--rw_images", help="Re-write encoded images if already exists.",
                        dest="rewrite_images", type=bool, default=False)
    parser.add_argument("--rw_model", help="Re-write model if already exists.",
                        dest="rewrite_model", type=bool, default=False)

    args = parser.parse_args()

    # Get path files
    print("Getting data...")
    train_images_path = os.path.join(args.train_path, "train")
    val_images_path = os.path.join(args.train_path, "validation")
    captions_path = os.path.join(args.train_path, "captions.txt")
    train_list = os.path.join(args.train_path, "train.txt")
    val_list = os.path.join(args.train_path, "validation.txt")

    # Get data from files
    data, max_length = dataFunctions.getData(captions_path)
    lex = dataFunctions.getLexicon(data)

    # Get images and captions arrays
    train_images_array, train_captions_array = dataFunctions.getDataArrays(
        data, train_list)

    val_images_array, val_captions_array = dataFunctions.getDataArrays(
        data, val_list)

    # Tokenize words using idxtoword and wordtoidx
    print("Tokenizing captions...")
    idxtoword, wordtoidx = dataFunctions.getTokenizers(lex)
    vocab_size = len(idxtoword) + 1

    # Get tokenized captions
    train_token_captions_array = dataFunctions.getTokensArrays(
        train_captions_array, wordtoidx)

    val_token_captions_array = dataFunctions.getTokensArrays(
        val_captions_array, wordtoidx)

    # Choose CNN model and get useful params
    print("Creating CNN model...")
    cnn_model = CNNModel(args.cnn_type)

    # Encode train images and save them
    if platform.system() == "Linux":
        name = args.train_path.split('/')[-1]
    elif platform.system() == "Windows":
        name = args.train_path.split('\\')[-1]

    train_images_save = os.path.join(
        "data", f'train_images_{name}_{cnn_model.get_output_dim()}.pk1')
    if not os.path.exists(train_images_save) or args.rewrite_images == True:
        print(f"Encoding images to {train_images_save}...")
        shape = (len(train_images_array), cnn_model.get_output_dim())
        train_encoded_images = np.zeros(shape=shape, dtype=np.float16)

        for i, img in enumerate(train_images_array):
            image_path = os.path.join(train_images_path, img)
            img = tensorflow.keras.preprocessing.image.load_img(
                image_path, target_size=(cnn_model.get_height(), cnn_model.get_width()))
            train_encoded_images[i] = cnn_model.encode_image(img)

        with open(train_images_save, 'wb') as f:
            pickle.dump(train_encoded_images, f)
            print("Saved encoded train images to disk")
    else:
        print(f"Loading images from {train_images_save}...")
        with open(train_images_save, 'rb') as f:
            train_encoded_images = pickle.load(f)

    # Encode val images and save them
    val_images_save = os.path.join(
        "data", f'val_images_{name}_{cnn_model.get_output_dim()}.pk1')
    if not os.path.exists(val_images_save) or args.rewrite_images == True:
        print(f"Encoding images to {val_images_save}...")
        shape = (len(val_images_array), cnn_model.get_output_dim())
        val_encoded_images = np.zeros(shape=shape, dtype=np.float16)

        for i, img in enumerate(val_images_array):
            image_path = os.path.join(val_images_path, img)
            img = tensorflow.keras.preprocessing.image.load_img(
                image_path, target_size=(cnn_model.get_height(), cnn_model.get_width()))
            val_encoded_images[i] = cnn_model.encode_image(img)

        with open(val_images_save, 'wb') as f:
            pickle.dump(val_encoded_images, f)
            print("Saved encoded validation images to disk")
    else:
        print(f"Loading images from {val_images_save}...")
        with open(val_images_save, 'rb') as f:
            val_encoded_images = pickle.load(f)

    # Load Embeddings if needed
    embedding_matrix = None
    if args.use_embedding == True:
        print("Loading embeddings...")
        embeddings = dataFunctions.getEmbeddings()
        embedding_matrix = dataFunctions.getEmbeddingMatrix(
            embeddings, vocab_size, wordtoidx)

    # Build model
    print("Building model...")
    rnn_model = RNNModel(args.rnn_type, args.neurons, vocab_size, max_length,
                         cnn_model.get_output_dim(), config.OPTIMIZER, embedding_matrix)

    rnn_model.build_model()
    rnn_model.compile_model()

    # Train model
    print("Training model...")
    model_save = os.path.join(
        args.model_path, f'model_{args.cnn_type}_{args.rnn_type}_{args.use_embedding}_{args.epochs}_{args.neurons}_{args.batch_size}.hdf5')
    if not os.path.exists(model_save) or args.rewrite_model == True:
        train_generator = rnn_model.create_generator(
            train_encoded_images, train_token_captions_array, args.batch_size)

        val_generator = rnn_model.create_generator(
            val_encoded_images, val_token_captions_array, len(val_encoded_images))

        early_stopping = tensorflow.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1, mode='min')

        reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=2, mode='min', verbose=1, min_lr=0.00001, cooldown=0, min_delta=0.00001)

        rnn_model.get_model().fit(train_generator, epochs=args.epochs, steps_per_epoch=(
            len(train_encoded_images) // args.batch_size), validation_data=val_generator,
            validation_steps=(len(val_encoded_images) // args.batch_size), verbose=2, callbacks=[early_stopping])

        if platform.system() == "Linux":
            plot_image(rnn_model.get_model().history.history['acc'], rnn_model.get_model(
            ).history.history['val_acc'], "Accuracy")
        elif platform.system() == "Windows":
            plot_image(rnn_model.get_model().history.history['accuracy'], rnn_model.get_model(
            ).history.history['val_accuracy'], "Accuracy")
        plot_image(rnn_model.get_model().history.history['loss'], rnn_model.get_model(
        ).history.history['val_loss'], "Loss")

        rnn_model.get_model().save(model_save)
        print(f'Saved model to {model_save}')

        with open(os.path.join(args.model_path, f'idxtoword_{args.cnn_type}_{args.rnn_type}_{args.use_embedding}_{args.epochs}_{args.neurons}_{args.batch_size}.pk1'), 'wb') as f:
            pickle.dump(idxtoword, f)
            print("Saved idxtoword to disk")

        with open(os.path.join(args.model_path, f'wordtoidx_{args.cnn_type}_{args.rnn_type}_{args.use_embedding}_{args.epochs}_{args.neurons}_{args.batch_size}.pk1'), 'wb') as f:
            pickle.dump(wordtoidx, f)
            print("Saved wordtoidx to disk")
    else:
        print(f'The model already exists at {model_save}')
