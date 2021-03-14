from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd





def conver_image_to_features(model, img_path, img_resize = (224, 224)):
    img = image.load_img(img_path, target_size=img_resize)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.reshape(-1) # convert to 1 dimention


def create_train_data_from_images(model, train_csv):
    df = pd.read_csv(train_csv)

    image_features = []

    for ind, row in df.iterrows():
        print("processing {} of {}".format(ind + 1, df.shape[0]))
        image_features.append(
            (conver_image_to_features(model, row["img1"]),
             conver_image_to_features(model, row["img2"]),
             row["same"]))

    res = pd.DataFrame({
        "x1": [x[0] for x in image_features],
        "x2": [x[1] for x in image_features],
        "y": [x[2] for x in image_features],
    })

    return res


def genearte_image_feature_file(csv_file, output):
    # last layer
    model = VGG16(weights="imagenet", include_top=False)

    res_df = create_train_data_from_images(model, csv_file)
    res_df.to_csv(output, index=False)

if __name__ == '__main__':
    genearte_image_feature_file("./min_index.csv", "min_feature.csv")
