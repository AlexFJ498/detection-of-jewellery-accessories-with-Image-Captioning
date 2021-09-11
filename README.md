# About the project
This project pretends to be a useful tool to create different deep learning architectures based on Image Captioning for detecting jewellery accesories. Specifically, the accesories need to be placed on the human body.

There are three datasets provided to create three different models, each one with a different level of caption. These levels are:

- Simple caption. With this level the model will be able to classify between four types of accesories: rings, earrings, pendants and bracelets.
- Medium caption. This level will provide the model with six types of accesories: gold rings, silver earrings, gold earrings, gold pendants, silver pendants and gold bracelets.
- Complete caption. With this level the model will provide the complete name of the accesory. For this reason, this level will only work with the same accesories used for training.

In this repository you will find useful scripts for training models, testing them, creating new data, and using a simple web page for a more visual way to test the results.

![Interface of web page](https://cdn.discordapp.com/attachments/886293563082678282/886293617751253002/pw_interface.png)

# Installation
## Download project
You can download the entire repository by using this command:

`$ git clone https://github.com/AlexFJ498/detection-of-jewellery-accessories-with-Image-Captioning.git`

You can also download the ZIP project manually from GitHub.

## Creating new environment
It is recommended to create a Python environment for the project. You can use `venv` by putting this command:

`$ python3 -m env env`

To activate the environment:

`$ source env/bin/activate`

## Install all libraries
You can install all the needed libraries by using this command:

`$ pip3 install -r requirements.txt`

## Creating useful directories
You'll need to create in the main directory the following folders:

- data. In this folder the script will save the encripted images of an specific dataset using a specific CNN.
- input. In this folder the datasets will be placed.
- models. In this folder the created models will be saved. It is recomended to create inside it a folder for each dataset with their same names.

## Download datasets
You can download the three datasets with the following command:

`$ python3 src/download.py`

You'll obtain three zip files, which you can unzip inside the input folder. The Accesorios_Genericos_bd dataset will be used for the simple captions. The Doñasol_bd dataset will be used for the medium captions. The Baquerizo_Joyeros_bd dataset will be used for the complete captions.

You should also download the pre-trained spanish word vectors to be able to use them during experiments using this [link](https://www.kaggle.com/rtatman/pretrained-word-vectors-for-spanish).

# Create and train models
You can train a model using train.py. For example, for training a model using Baquerizo Joyeros dataset:

`$ python3 src/train.py --train_path input/Baquerizo_Joyeros_bd --model_path models/Baquerizo_Joyeros_bd --cnn inception --rnn gru --neurons 256 --epochs 50 --batch_size 8 --use_embedding True`

You can read the full description of every parameter inside the script.

Once the script is finished, you'll be able to see some graphics about the accuracy and loss values obtained during the training process.

# Test the model
For testing the models with the test set you can use test.py. Following the previous example:

`$ python3 src/test.py test_path input/Baquerizo_Joyeros_bd --model models/Baquerizo_Joyeros_bd/model_inception_gru_True_50_256_8.hdf5`

The script will print the obtained results for each sample, and will end showing a confusion matrix if it's possible (only with the first two levels).

# Generate new data using Image Data Augmentation
Keras provides useful tools to create new images using Image Data Augmentation. You can execute dataAugmentation.py for create a specific number of images for each sample:

`$ python3 src/dataAugmentation.py --images_path input/Baquerizo_Joyeros_bd/train --save_path temp_folder --num 5`

This example will generate 5 new pics for each image existing in the train folder. 
You can add the generated images with addAugmentation.py:

`$ python3 src/addAugmentation.py --dataset input/Baquerizo_Joyeros_bd --data_path temp_folder --type train`

This example will move the generated images to the train folder, and will update the captions files correctly.

# Using web page
The web page has been created using the framework called Flask. First of all you'll need to open app.py script and change the NAME_1, NAME_2 and NAME_3 variables, You'll provide the best models obtained during experiments. The first name refers to the simple caption. The second one refers to the medium caption. The last one is the complete caption. You need to provide the name without the "model_" part and the final extension (For example: `NAME_1 = 'inception_gru_True_50_256_8'`).

Now you can activate the web page using app.py script:

`$ python3 src/app.py`

Now enter to localhost:5000 using any web explorer and you'll be able to use the interface.

You'll be able to upload any local image and choose any caption level. Push the button and the result will appear. You can repeat the process every time you want.

![Choosing level on web page](https://cdn.discordapp.com/attachments/886293563082678282/886293712240533604/pw_level4.png)

![Results interface](https://cdn.discordapp.com/attachments/886293563082678282/886293815676264498/pw_final2.png)
