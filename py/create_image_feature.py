from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from time import gmtime, strftime, strptime, localtime
import numpy as np
import pandas as pd
from tqdm import tqdm
import regex as re
import os
from os.path import basename
import glob


IMAGES_FOLDERS = [
    "../../data/train_1",
    "../../data/train_2",
    "../../data/train_3",
    "../../data/train_4",
    "../../data/train_5",
    "../../data/train_6",
    "../../data/train_7",
    "../../data/train_8",
    "../../data/train_9",
    "../../data/test",
]

TARGET_AUTHOR = [
    "Zdislav Beksinski",
    "Ivan Aivazovsky",
    "Pablo picasso",
    "Ilya Repin",
    "Ivan shishkin",
    "Pierre-Auguste Renoir",
    "ALbrecht Durer",
    "John Singer sargent",
    "Gustave dore",
    "Marc Chagall",
    "Giovanni Battista Piranesi",
    "Rembrandt",
    "Martiros saryan",
    "Paul Cezanne",
    "Camille Pissarro",
]

AUTHOR_LIST = "./all_data_info.csv"


#======================================================================

def gen_img_todo_list(author_list):
    df = pd.read_csv(author_list)
    df = df[df.artist.isin(TARGET_AUTHOR)][['artist', 'new_filename']]
    return df

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

def gen_output_folder(output, modle_name, layer_name, batch_size, img_size):

    output_sub_folder = "{}/{}_{}_{}_{}_{}/".format(output,
                           modle_name,
                           layer_name,
                           img_size,
                           batch_size,
                           strftime("%Y_%m_%d_%H_%M_%S", localtime()))
    create_folder(output_sub_folder)

    return output_sub_folder


def gen_model():

    # last layer
    return VGG16(weights="imagenet", include_top=False)


def __gen_feature(model, img, resize):
    img = image.load_img(img, target_size=resize)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.reshape(-1)


def gather_avaiable_images(target_folder):
    """
    :param target_folder:
    :return: image file name list
    """
    targets = []

    for target in target_folder:
        imges = glob.glob(target + u"/*.jpg")
        targets.extend([(target, basename(x)) for x in imges])

    return pd.DataFrame(data=targets, columns=["path", "filename"])



def find_img_path(img_name, avail_img_df):

    res = avail_img_df[avail_img_df.filename == img_name]

    if res.shape[0] == 0:
        return ""
    else:
        return "{}/{}".format(res.iloc[0]['path'], img_name)


def save_to_batch_file(image_features, author_name, image_path, out_path, batch_index):
    out_df = pd.DataFrame().from_records(image_features)
    out_df["artist"] = author_name
    out_df["image"] = image_path
    out_file = '{}/{}.csv'.format(out_path, batch_index)
    out_df.to_csv(out_file, index=False, header=False, float_format='%.3f')


def gen_features(model, output, aut_df, batch_size, img_size):

    out_path = gen_output_folder(output,
                                 model.name,
                                 re.sub(r"[/:_]", "", model.output.name),
                                 re.sub(r"[\(,\)]", "_", str(img_size)),
                                 batch_size)

    out_path = out_path.replace(" ", "")

    image_features = []
    author_name = []
    image_path = []
    batch_index = 0
    image_index = 0

    avail_img_df = gather_avaiable_images(IMAGES_FOLDERS)

    for ind, row in tqdm(aut_df.iterrows(), total=aut_df.shape[0]):
        try:
            img_path = find_img_path(row['new_filename'], avail_img_df)

            if img_path == "":
                print("Can't find file {}".format(row['new_filename']))
                continue

            feature = __gen_feature(model, img_path, img_size)

            image_features.append(feature)
            author_name.append(row["artist"])
            image_path.append(row['new_filename'])

            image_index += 1

            # Batch process
            if image_index == batch_size:
                image_index = 0
                save_to_batch_file(image_features, author_name, image_path, out_path, batch_index)

                image_features = []
                author_name = []
                image_path = []
                batch_index += 1
        except:
            print("Error img {}".format(row['new_filename']))

    if len(image_features) != 0:
        save_to_batch_file(image_features, author_name, image_path, out_path, batch_index)


if __name__ == '__main__':

    OUTPUT = "./output"
    IMG_RESIZE = (224,224)
    BATCH_SIZE = 200
    #===================

    model = gen_model()
    author_df = gen_img_todo_list(AUTHOR_LIST)
    gen_features(model, OUTPUT, author_df, BATCH_SIZE, IMG_RESIZE)

















