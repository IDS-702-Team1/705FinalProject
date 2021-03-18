from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from time import gmtime, strftime, strptime, localtime
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
import multiprocessing as mp
import regex as re
import os


def __gen_feature(model, img, resize):
    img = image.load_img(img, target_size=resize)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.reshape(-1)

def absolute_diff_feature(model, img_1, img_2, img_resize = (224, 224)):
    img1 = __gen_feature(model, img_1, img_resize)
    img2 = __gen_feature(model, img_2, img_resize)
    return img1 - img2

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


def create_features_dataframe(model,
                              train_csv,
                              output,
                              batch=200,
                              img_resize=(224, 224),
                              feature_diff='abs'):

    reader = pd.read_csv(train_csv, chunksize=batch)
    index = 0

    output_sub_folder = "{}/{}_{}_{}_{}_{}_{}/".format(output,
                                                 model.name,
                                                 re.sub(r"[/:_]", "", model.output.name),
                                                 feature_diff,
                                                 img_resize,
                                                 batch,
                                                 strftime("%Y_%m_%d_%H_%M_%S", localtime()))

    create_folder(output_sub_folder)

    for df in reader:

        image_features = []
        index += 1

        print("Processing {} chunck [size: {}]".format(index, batch))

        for ind, row in tqdm(df.iterrows(), total=df.shape[0]):
            try:
                feature = absolute_diff_feature(model, row["img1"], row["img2"] )
                image_features.append(np.append(feature, row["same"]))
            except:
                print("Error img {} {}".format(row["img1"], row["img2"]))

        out_df = pd.DataFrame().from_records(image_features)

        out_file = '{}/{}.csv'.format(output_sub_folder, index)
        out_df.to_csv(out_file, index=False, header=False, float_format='%.3f')


def genearte_image_feature_file(input, output, decimal_number=3):

    # last layer
    model = VGG16(weights="imagenet", include_top=False)
    create_features_dataframe(model, input, output, batch=200)


if __name__ == '__main__':
    genearte_image_feature_file("./train_10000.csv", "./output")
