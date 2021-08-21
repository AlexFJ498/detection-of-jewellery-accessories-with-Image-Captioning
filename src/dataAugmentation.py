from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import expand_dims
import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to create new images from dataset using Data Augmentation')

    parser.add_argument('--images_path', help="Images path",
                        dest="images_path", type=str, required=True)
    parser.add_argument("--save_path", help="Save path",
                        dest="save_path", type=str, required=True)
    parser.add_argument("--num", help="Number of new images for each one",
                        dest="num", type=int, required=True)

    args = parser.parse_args()

    datagen = ImageDataGenerator(rotation_range=90, width_shift_range=0.3,
                                 height_shift_range=0.3, shear_range=0.15,
                                 zoom_range=0.05, channel_shift_range=10, horizontal_flip=True,
                                 vertical_flip=True, brightness_range=[0.2, 1.0])

    for image_path in os.listdir(args.images_path):
        image = load_img(os.path.join(args.images_path, image_path))
        image = img_to_array(image)
        image = expand_dims(image, axis=0)

        datagen.fit(image)

        number_of_images = args.num
        for x, val in datagen.flow(image, image,
                                   save_to_dir=args.save_path,
                                   save_prefix=image_path.split('.')[0],
                                   save_format='jpg',
                                   batch_size=9):

            number_of_images -= 1
            if number_of_images == 0:
                break

    print("Done!")
