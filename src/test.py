import dataFunctions
from modelFunctions import CNNModel, RNNModel
from tensorflow import keras
import tensorflow.keras.preprocessing.image
import platform
import pickle
import numpy as np
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    parser = argparse.ArgumentParser(
        description='Script to test Image Captioning models')

    parser.add_argument('--model', help="Model to test",
                        dest="model_name", type=str, required=True)
    parser.add_argument('--test_path', help="Path to test data",
                        dest="test_path", type=str, required=True)
    parser.add_argument("--rw_images", help="Re-write encoded images if already exists.",
                        dest="rewrite_images", type=bool, default=False)

    args = parser.parse_args()

    # Get path files
    print("Getting data...")
    images_path = os.path.join(args.test_path, "test")
    captions_path = os.path.join(args.test_path, "captions.txt")
    test_list = os.path.join(args.test_path, 'test.txt')

    # Get data from files
    data, max_length = dataFunctions.getData(captions_path)
    lex = dataFunctions.getLexicon(data)

    # Get images and captions arrays
    images_array, captions_array = dataFunctions.getDataArrays(data, test_list)

    # Tokenize words using idxtoword and wordtoidx
    print("Tokenizing captions...")
    idxtoword, wordtoidx = dataFunctions.getTokenizers(lex)
    vocab_size = len(idxtoword) + 1

    # Get model info
    if platform.system() == "Linux":
        cnn_type = args.model_name.split('/')[-1].split('_')[1]
        rnn_type = args.model_name.split('/')[-1].split('_')[2]
        use_embedding = args.model_name.split('/')[-1].split('_')[3]
        epochs = args.model_name.split('/')[-1].split('_')[4]
        neurons = args.model_name.split('/')[-1].split('_')[5].split('.')[0]
        name = args.test_path.split('/')[-1]
    elif platform.system() == "Windows":
        cnn_type = args.model_name.split('\\')[-1].split('_')[1]
        rnn_type = args.model_name.split('\\')[-1].split('_')[2]
        use_embedding = args.model_name.split('\\')[-1].split('_')[3]
        epochs = args.model_name.split('\\')[-1].split('_')[4]
        neurons = args.model_name.split('\\')[-1].split('_')[5].split('.')[0]
        name = args.test_path.split('\\')[-1]

    # Choose CNN model and get useful params
    print("Creating CNN model...")
    cnn_model = CNNModel(cnn_type)

    # Encode images and save them
    images_save = os.path.join(
        "data", f'test_images_{name}_{cnn_model.get_output_dim()}.pk1')
    if not os.path.exists(images_save) or args.rewrite_images == True:
        print(f"Encoding images to {images_save}...")
        shape = (len(images_array), cnn_model.get_output_dim())
        encoded_images = np.zeros(shape=shape, dtype=np.float16)

        for i, img in enumerate(images_array):
            image_path = os.path.join(images_path, img)
            img = tensorflow.keras.preprocessing.image.load_img(
                image_path, target_size=(cnn_model.get_height(), cnn_model.get_width()))
            encoded_images[i] = cnn_model.encode_image(img)

        with open(images_save, 'wb') as f:
            pickle.dump(encoded_images, f)
            print("Saved encoded images to disk")
    else:
        print(f"Loading images from {images_save}...")
        with open(images_save, 'rb') as f:
            encoded_images = pickle.load(f)

    # Get number of accesories are of each type
    num_gargantillas, num_pendientes, num_anillos, num_pulseras = getNumAccesories(
        captions_array)

    print("Loading model...: ")
    rnn_model = RNNModel(max_length=max_length)
    rnn_model.set_model(keras.models.load_model(args.model_name))
    print("Model loaded")

    with open(os.path.join("models", f"models_{name}", f'idxtoword_{cnn_type}_{rnn_type}_{use_embedding}_{epochs}_{neurons}.pk1'), 'rb') as f:
        idxtoword = pickle.load(f)
        print("Loaded idxtoword from disk")

    with open(os.path.join("models", f"models_{name}", f'wordtoidx_{cnn_type}_{rnn_type}_{use_embedding}_{epochs}_{neurons}.pk1'), 'rb') as f:
        wordtoidx = pickle.load(f)
        print("Loaded wordtoidx from disk")
        print("================================")

    ccr = 0
    ccr_collares = 0
    ccr_pendientes = 0
    ccr_anillos = 0
    ccr_pulseras = 0

    for i in range(len(images_array)):
        image = images_array[i]
        image_encoded = encoded_images[i]

        print("Expected image: ", image)
        print("Expected caption:", data[image])
        if rnn_type == "lstm":
            caption = rnn_model.generate_caption_LSTM(
                image_encoded, wordtoidx, idxtoword)
        elif rnn_type == "gru":
            caption = rnn_model.generate_caption_GRU(
                image, wordtoidx, idxtoword, cnn_model, images_path)

        print("Obtained caption:", caption)

        predicted_image, img = searchCaption(caption, data)

        if predicted_image:
            if caption == data[image]:
                ccr += 1

                if "Gargantilla" in caption or "Colgante" in caption:
                    ccr_collares += 1

                elif "Pendiente" in caption or "Pendientes" in caption:
                    ccr_pendientes += 1

                elif "Anillo" in caption or "Sortija" in caption:
                    ccr_anillos += 1

                elif "Pulsera" in caption:
                    ccr_pulseras += 1
        else:
            print("The obtained caption does not exist in the dataset")

        print("================================")

    print(ccr)
    print(ccr_collares)
    print(ccr_pendientes)
    print(ccr_anillos)
    print(ccr_pulseras)

    ccr = obtainCCR(ccr, len(images_array))
    ccr_gargantillas = obtainCCR(ccr_collares, num_gargantillas)
    ccr_pendientes = obtainCCR(ccr_pendientes, num_pendientes)
    ccr_anillos = obtainCCR(ccr_anillos, num_anillos)
    ccr_pulseras = obtainCCR(ccr_pulseras, num_pulseras)

    print("CCR =", round(ccr, 2), "%")
    print("CCR Gargantillas=", round(ccr_gargantillas, 2), "%")
    print("CCR Pendientes=", round(ccr_pendientes, 2), "%")
    print("CCR Anillos=", round(ccr_anillos, 2), "%")
    print("CCR Pulseras=", round(ccr_pulseras, 2), "%")