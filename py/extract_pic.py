import shutil
import os
import pandas as pd




def create_folder(path):

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


def prepare_output_folder(output_folder):

    pos_folder = "{}/pos".format(output_folder)
    neg_folder = "{}/neg".format(output_folder)

    create_folder(pos_folder)
    create_folder(neg_folder)

    return (pos_folder, neg_folder)


def extract_image_pair(train_file, output_folder):

    pos_folder, neg_folder = prepare_output_folder(output_folder)


    df = pd.read_csv(train_file)

    for ind, row in df.iterrows():

        print("processing {} of {}".format(ind, df.shape[0]))

        if row["same"] == 1:
            folder = pos_folder
        elif row["same"] == 0:
            folder = neg_folder
        else:
            raise Exception("Unknown same value")


        shutil.copyfile(row["img1"], "{}/{}_A.jpg".format(folder, ind))
        shutil.copyfile(row["img2"], "{}/{}_B.jpg".format(folder, ind))



if __name__ == '__main__':

    extract_image_pair("./min_index.csv", "./min")