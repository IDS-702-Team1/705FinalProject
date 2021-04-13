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
from multiprocessing import Pool, cpu_count



TARGET_AUTHOR = [
    "Zd# islav Beksinski",
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

def gen_img_todo_list_by_author(author_list):
    df = pd.read_csv(author_list)
    df = df[df.artist.isin(TARGET_AUTHOR)][['artist', 'new_filename', 'genre']]
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


def gen_model(model = "vgg16", layer = "top"):

    if model == "vgg16" and layer == 'top':
        return VGG16(weights="imagenet", include_top=False)
    elif model == "vgg16" and layer == 'layer1':
        base_model = VGG16(weights="imagenet")
        model = Model(inputs = base_model.input, outputs = base_model.get_layer('block1_conv2').output)
        return model
    else:
        print("Unsupport Model type")
        raise Exception()

def __gen_feature(model, img, resize):
    img = image.load_img(img, target_size=resize)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.reshape(-1)


def gather_avaiable_images(target_folder_list):
    """
    :param target_folder:
    :return: image file name list
    """
    targets = []

    for target in target_folder_list:
        imges = glob.glob(target + u"/*.jpg")
        targets.extend([(target, basename(x)) for x in imges])

    df = pd.DataFrame({
        "filename": [x[1] for x in targets],
        "path": [x[0] for x in targets],
    })

    return df



def find_img_path(img_name, avail_img_df):

    res = avail_img_df[avail_img_df.filename == img_name]

    if res.shape[0] == 0:
        return ""
    else:
        return "{}/{}".format(res.iloc[0]['path'], img_name)




def save_to_batch_file(image_features, author_name, image_path, genre_list, out_path, batch_index):
    out_df = pd.DataFrame().from_records(image_features)
    out_df["artist"] = author_name
    out_df["image"] = image_path
    out_df["genre"] = genre_list

    out_file = '{}/{}.csv'.format(out_path, batch_index)
    out_df.to_csv(out_file, index=False, header=False, float_format='%.3f')


def gen_features(model, output, aut_df, batch_size, img_size, jidx=0):

    out_path = gen_output_folder(output,
                                 model.name,
                                 re.sub(r"[/:_]", "", model.output.name),
                                 re.sub(r"[\(,\)]", "_", str(img_size)),
                                 batch_size)

    out_path = out_path.replace(" ", "").replace("__", "_")

    create_folder(out_path)

    image_features = []
    author_name = []
    image_path = []
    genre_list = []
    batch_index = 1000 * jidx
    image_index = 0

    avail_img_df = gather_avaiable_images(IMAGES_FOLDERS)


    for ind, row in tqdm(aut_df.iterrows(), total=aut_df.shape[0], desc="job {}".format(jidx)):
        try:
            img_path = find_img_path(row['new_filename'], avail_img_df)

            if img_path == "":
                print("Can't find file {}".format(row['new_filename']))
                continue

            #print("Processing " + img_path)

            feature = __gen_feature(model, img_path, img_size)

            image_features.append(feature)
            author_name.append(row["artist"])
            image_path.append(row['new_filename'])
            genre_list.append(row['genre'])

            image_index += 1

            # Batch process
            if image_index == batch_size:
                image_index = 0
                save_to_batch_file(image_features, author_name, image_path, genre_list, out_path, batch_index)

                image_features = []
                author_name = []
                image_path = []
                genre_list = []
                batch_index += 1
        except:
            print("Error img {}".format(row['new_filename']))

    if len(image_features) != 0:
        save_to_batch_file(image_features, author_name, image_path, genre_list, out_path, batch_index)

def __gen_features(args):
    output, aut_df, batch_size, img_size, jidx, model_type, model_layer = args
    model = gen_model(model_type, model_layer)
    gen_features(model, output, aut_df, batch_size, img_size, jidx)

def split_dataframe_into_n_sub_dataframe(df, bin_number):
    '''
    Split An Dataframe to some sub dataframe,

    from multiprocessing import Pool

    p = Pool(job)
    task_df = split_dataframe_into_n_sub_dataframe(df, job)
    p.map(threadMethod, [(subdf, args) for subdf in task_df])

    :param df:  Original dataframe
    :param bin_number:  Split to how many sub arrays
    :return: grouped
    '''
    grouped = df.groupby(df.index % bin_number)
    return [grouped.get_group(i) for i in range(0, len(grouped)) ]




def multi_process_gen_features(output, aut_df, batch_size, img_size, job=1, model_type = 'vgg16', model_layer = "top"):

    p = Pool(job)
    task_df = split_dataframe_into_n_sub_dataframe(aut_df, job)
    p.map(__gen_features, [(output, subdf, batch_size, img_size, jidx, model_type, model_layer) for jidx, subdf in enumerate(task_df)])


def gen_img_todo_list_by_folder(author_list, target_images):
    df = pd.read_csv(author_list)
    df = df[df.new_filename.isin(target_images.filename)][['artist', 'new_filename', 'genre']]
    return df


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


def gen_vgg16_block5(output, image_size, batch_size, job):
    # VGG16 top-layer
    target_images = gather_avaiable_images(IMAGES_FOLDERS)
    author_df = gen_img_todo_list_by_folder(AUTHOR_LIST, target_images)
    multi_process_gen_features(output, author_df, batch_size, image_size, job=job, model_type='vgg16', model_layer="top")

def gen_vgg16_block1(output, image_size, batch_size, job):
    # VGG16 top-layer
    target_images = gather_avaiable_images(IMAGES_FOLDERS)
    author_df = gen_img_todo_list_by_folder(AUTHOR_LIST, target_images)
    multi_process_gen_features(output, author_df, batch_size, image_size, job=job, model_type='vgg16', model_layer="layer1")

if __name__ == '__main__':

    OUTPUT = "./output"
    IMG_RESIZE = (224,224)
    BATCH_SIZE = 200

    job = cpu_count()
    #===================

    gen_vgg16_block1(OUTPUT, IMG_RESIZE, BATCH_SIZE, job)











