import os
import sys
import argparse


def read_captions(captions_file):
    captions_dict = dict()
    with open(captions_file, 'r') as f:
        for line in f.read().split('\n'):
            line = line.split()
            image = line[0]
            caption = line[1:]

            captions_dict[image] = ' '.join(caption)

    return captions_dict


def add_augmentation(data_path, caption_path, captions_dict, type):
    for image in os.listdir(data_path):
        name = image.split("_")[0] + ".jpg"
        with open(caption_path, 'a') as f:
            if name in captions_dict.keys():
                caption = captions_dict[name]

            else:
                caption = "Manual caption needed!"

            f.write(f"{image} \t\t {caption}\n")

        with open(type, 'a') as f:
            f.write(f"{image}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', help="Path of the dataset",
                        dest="dataset", type=str, required=True)
    parser.add_argument('--data_path', help="Path of new images",
                        dest="data_path", type=str, required=True)
    parser.add_argument('--type', help="Type of images that are going to be added (Train, validation or test)",
                        dest="type", type=str, required=True)

    args = parser.parse_args()

    captions = os.path.join(args.dataset, 'captions.txt')
    captions_dict = read_captions(captions)

    if args.type == 'train':
        type = os.path.join(args.dataset, 'train.txt')

    elif args.type == 'validation':
        type = os.path.join(args.dataset, 'validation.txt')

    elif args.type == 'test':
        type = os.path.join(args.dataset, 'test.txt')

    else:
        print("Type not valid")
        sys.exit(1)

    add_augmentation(args.data_path, captions, captions_dict, type)
    print("Added new data!")
