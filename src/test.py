import argparse
import numpy as np
import pickle
import platform
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.preprocessing.image
from tensorflow import keras
from modelFunctions import CNNModel, RNNModel
import dataFunctions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def searchCaption(caption, data):
    predicted_image = False
    for img, cap in data.items():
        if caption == cap:
            predicted_image = True
            break

    return predicted_image, img


def obtain_CCR(ccr, total):
    ccr = ccr / total
    ccr = ccr * 100
    return ccr


def get_num_accesories_4(captions_array):
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


def get_num_accesories_6(captions_array):
    num_pendientes_plata = 0
    num_pendientes_oro = 0
    num_colgantes_plata = 0
    num_colgantes_oro = 0
    num_anillos_oro = 0
    num_pulseras_oro = 0

    for caption in captions_array:
        word = caption.split()[1]
        if word == 'Pendiente' or word == 'Pendientes' and "plata" in caption:
            num_pendientes_plata += 1

        elif word == 'Pendiente' or word == 'Pendientes' and "oro" in caption:
            num_pendientes_oro += 1

        elif word == 'Colgante' and "plata" in caption:
            num_colgantes_plata += 1

        elif word == 'Colgante' and "oro" in caption:
            num_colgantes_oro += 1

        elif word == 'Anillo' and "oro" in caption:
            num_anillos_oro += 1

        elif word == 'Pulsera' and "oro" in caption:
            num_pulseras_oro += 1

    return num_pendientes_plata, num_pendientes_oro, num_colgantes_plata, num_colgantes_oro, num_anillos_oro, num_pulseras_oro


def get_CCR_4(caption, ccr_collares, ccr_pendientes, ccr_anillos, ccr_pulseras):
    if "Gargantilla" in caption or "Colgante" in caption:
        ccr_collares += 1

    elif "Pendiente" in caption or "Pendientes" in caption:
        ccr_pendientes += 1

    elif "Anillo" in caption or "Sortija" in caption:
        ccr_anillos += 1

    elif "Pulsera" in caption:
        ccr_pulseras += 1

    return ccr_collares, ccr_pendientes, ccr_anillos, ccr_pulseras


def get_CCR_6(caption, ccr_pendientes_plata, ccr_pendientes_oro, ccr_colgantes_plata,
              ccr_colgantes_oro, ccr_anillos_oro, ccr_pulseras_oro):

    if ("Pendiente" in caption or "Pendientes" in caption) and "plata" in caption:
        ccr_pendientes_plata += 1

    elif ("Pendiente" in caption or "Pendientes" in caption) and "oro" in caption:
        ccr_pendientes_oro += 1

    elif "Colgante" in caption and "plata" in caption:
        ccr_colgantes_plata += 1

    elif "Colgante" in caption and "oro" in caption:
        ccr_colgantes_oro += 1

    elif "Anillo" in caption and "oro" in caption:
        ccr_anillos_oro += 1

    elif "Pulsera" in caption and "oro" in caption:
        ccr_pulseras_oro += 1

    return ccr_pendientes_plata, ccr_pendientes_oro, ccr_colgantes_plata, ccr_colgantes_oro, ccr_anillos_oro, ccr_pulseras_oro


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
        neurons = args.model_name.split('/')[-1].split('_')[5]
        batch_size = args.model_name.split('/')[-1].split('_')[6].split('.')[0]
        name = args.test_path.split('/')[-1]
    elif platform.system() == "Windows":
        cnn_type = args.model_name.split('\\')[-1].split('_')[1]
        rnn_type = args.model_name.split('\\')[-1].split('_')[2]
        use_embedding = args.model_name.split('\\')[-1].split('_')[3]
        epochs = args.model_name.split('\\')[-1].split('_')[4]
        neurons = args.model_name.split('\\')[-1].split('_')[5]
        batch_size = args.model_name.split(
            '\\')[-1].split('_')[6].split('.')[0]
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
            print("Saved encoded test images to disk")
    else:
        print(f"Loading images from {images_save}...")
        with open(images_save, 'rb') as f:
            encoded_images = pickle.load(f)

    # Get number of accesories are of each type
    if name == "Donasol_bd":
        num_pendientes_plata, num_pendientes_oro, num_colgantes_plata, num_colgantes_oro, num_anillos_oro, num_pulseras_oro = get_num_accesories_6(
            captions_array)
    else:
        num_gargantillas, num_pendientes, num_anillos, num_pulseras = get_num_accesories_4(
            captions_array)

    print("Loading model...: ")
    rnn_model = RNNModel(max_length=max_length)
    rnn_model.set_model(keras.models.load_model(args.model_name))
    print("Model loaded")

    with open(os.path.join("models", name, f'idxtoword_{cnn_type}_{rnn_type}_{use_embedding}_{epochs}_{neurons}_{batch_size}.pk1'), 'rb') as f:
        idxtoword = pickle.load(f)
        print("Loaded idxtoword from disk")

    with open(os.path.join("models", name, f'wordtoidx_{cnn_type}_{rnn_type}_{use_embedding}_{epochs}_{neurons}_{batch_size}.pk1'), 'rb') as f:
        wordtoidx = pickle.load(f)
        print("Loaded wordtoidx from disk")
        print("================================")

    ccr = 0

    ccr_collares = 0
    ccr_pendientes = 0
    ccr_anillos = 0
    ccr_pulseras = 0

    ccr_pendientes_plata = 0
    ccr_pendientes_oro = 0
    ccr_colgantes_plata = 0
    ccr_colgantes_oro = 0
    ccr_anillos_oro = 0
    ccr_pulseras_oro = 0

    expected_array = []
    obtained_array = []
    matrix_labels = []

    for i in range(len(images_array)):
        image = images_array[i]
        image_encoded = encoded_images[i]

        print("Expected image: ", image)
        print("Expected caption:", data[image])
        caption = rnn_model.generate_caption(
            image_encoded, wordtoidx, idxtoword)

        print("Obtained caption:", caption)

        expected_array.append(data[image])
        obtained_array.append(caption)

        if data[image] not in matrix_labels:
            matrix_labels.append(data[image])

        predicted_image, img = searchCaption(caption, data)

        if predicted_image:
            if caption == data[image]:
                ccr += 1

                if name == "Donasol_bd":
                    ccr_pendientes_plata, ccr_pendientes_oro, ccr_colgantes_plata, ccr_colgantes_oro, ccr_anillos_oro, ccr_pulseras_oro = get_CCR_6(caption, ccr_pendientes_plata, ccr_pendientes_oro, ccr_colgantes_plata,
                                                                                                                                                    ccr_colgantes_oro, ccr_anillos_oro, ccr_pulseras_oro)

                else:
                    ccr_collares, ccr_pendientes, ccr_anillos, ccr_pulseras = get_CCR_4(caption, ccr_collares, ccr_pendientes,
                                                                                        ccr_anillos, ccr_pulseras)

        else:
            print("The obtained caption does not exist in the dataset")

        print("================================")

    print(ccr)
    ccr = obtain_CCR(ccr, len(images_array))
    print("CCR =", round(ccr, 2), "%")

    if name == "Donasol_bd":

        ccr_pendientes_plata = obtain_CCR(
            ccr_pendientes_plata, num_pendientes_plata)
        ccr_pendientes_oro = obtain_CCR(ccr_pendientes_oro, num_pendientes_oro)
        ccr_colgantes_plata = obtain_CCR(
            ccr_colgantes_plata, num_colgantes_plata)
        ccr_colgantes_oro = obtain_CCR(ccr_colgantes_oro, num_colgantes_oro)
        ccr_anillos_oro = obtain_CCR(ccr_anillos_oro, num_anillos_oro)
        ccr_pulseras_oro = obtain_CCR(ccr_pulseras_oro, num_pulseras_oro)

        print("CCR Pendientes Plata =", round(ccr_pendientes_plata, 2), "%")
        print("CCR Pendientes Oro =", round(ccr_pendientes_oro, 2), "%")
        print("CCR Colgantes Plata =", round(ccr_colgantes_plata, 2), "%")
        print("CCR Colgantes Oro =", round(ccr_colgantes_oro, 2), "%")
        print("CCR Anillos Oro =", round(ccr_anillos_oro, 2), "%")
        print("CCR Pulseras Oro =", round(ccr_pulseras_oro, 2), "%")

    else:

        ccr_colgantes = obtain_CCR(ccr_collares, num_gargantillas)
        ccr_pendientes = obtain_CCR(ccr_pendientes, num_pendientes)
        ccr_anillos = obtain_CCR(ccr_anillos, num_anillos)
        ccr_pulseras = obtain_CCR(ccr_pulseras, num_pulseras)

        print("CCR Collares =", round(ccr_colgantes, 2), "%")
        print("CCR Pendientes =", round(ccr_pendientes, 2), "%")
        print("CCR Anillos =", round(ccr_anillos, 2), "%")
        print("CCR Pulseras =", round(ccr_pulseras, 2), "%")

    print("---------------------")
    if name == "Donasol_bd" or name == "Accesorios_Genericos_bd":
        print("CONFUSION MATRIX")
        matrix = confusion_matrix(
            expected_array, obtained_array, matrix_labels)

        print(matrix)
        print(matrix_labels)

        df_cm = pd.DataFrame(matrix, matrix_labels, matrix_labels)
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})

        plt.show()
