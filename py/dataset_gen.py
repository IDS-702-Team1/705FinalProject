from os.path import basename
import glob
import pandas as pd
from sklearn.utils import shuffle

IMAGES = [
    "../../data/train_1",
    "../../data/train_2"
]

TRAIN_INFO = "../../data/train_info.csv"

RANDOM_SEED = 6666



def gather_avaiable_images(target_folder:list):
    """
    :param target_folder:
    :return: image file name list
    """
    targets = []

    for target in target_folder:
        imges = glob.glob(target + u"/*.jpg")
        targets.extend([(target, basename(x)) for x in imges])

    return pd.DataFrame(data=targets, columns=["path", "filename"])


def generate_sample(r1, r2, is_same):

    return ("{}/{}".format(r1["path"], r1["filename"]),
            "{}/{}".format(r2["path"], r2["filename"]),
            int(is_same))



def generate_dataset(filename, train_info, size = 10000, possitive=0.5):

    assert (possitive > 0) and (possitive < 1)

    df_avail = gather_avaiable_images(IMAGES)
    df_train_info = pd.read_csv(train_info)[["filename", "artist"]]

    df = df_train_info.merge(df_avail, on="filename")

    count_table = df.groupby(["artist"]).size().reset_index()
    count_table.columns = ['artist', 'size']
    count_table.sort_values('size', ascending=False, inplace=True)

    positive_size = int(size * possitive)
    negative_size = int(size * (1- possitive))

    samples = []

    #Choose positive sample
    positive_artist = count_table[count_table["size"] > 1]\
                        .sample(positive_size, replace=True, random_state=1234)["artist"]

    for artist in positive_artist:
        select = df[df.artist == artist].\
            sample(2, replace=False, random_state=1234)
        samples.append(generate_sample(select.iloc[0], select.iloc[1], True))

    #Choose negative sample

    for i in range(0, negative_size):
        negative_artist = count_table.sample(2, replace=True, random_state=1234)["artist"]
        select1 = df[df.artist == negative_artist.iloc[0]]. \
            sample(1, replace=False, random_state=1234)
        select2 = df[df.artist == negative_artist.iloc[1]]. \
            sample(1, replace=False, random_state=1234)

        samples.append(generate_sample(select1.iloc[0], select2.iloc[0], False))

    final_df = pd.DataFrame(samples, columns=["img1", "img2", "same"])

    final_df = shuffle(final_df)

    final_df.to_csv(filename, index=False)

if __name__ == '__main__':

    generate_dataset("train_index.csv", TRAIN_INFO, size = 2000)
    generate_dataset("test_index.csv", TRAIN_INFO, size=300)