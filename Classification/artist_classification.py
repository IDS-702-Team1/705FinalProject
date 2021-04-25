import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import plotting_metrics
import numpy as np




def model_metrics(model,X_test,y_test):

    # make predictions for test data
    y_pred_test = model.predict(X_test)

    #Model Classification Report
    cr_vgg_style=classification_report(y_test, y_pred_test)

    #Model Confusion Matrix
    cm_vgg_style=confusion_matrix(y_test,y_pred_test)


    #Plot the Confusion Matrix (Heat Map)
    if model_counter==1:

        title="Heat Map of VGG16 Painting Artist Classification"
    else:
        title="Heat Map of ResNet50 Painting Artist Classification"

    list_labels=['Albrecht Durer', 'Camille Pissarro','Gustave Dore','Ivan Aivazovsky', 'Ivan Shishkin', 'Marc Chagall', 'Martiros Saryan', 'Pablo Picasso',
    'Pierre-Auguste Renoir', 'Rembrandt', 'Vincent van Gogh']
    plotting_metrics.plot_confusion_matrix(cm_vgg_style,title,list_labels)

    #Plot the ROC Curve
    plotting_metrics.plot_roc(model,X_test,y_test)




def model(X_train, X_test, y_train, y_test):

    #XGB Model
    model = XGBClassifier(n_estimators=250, n_jobs=-1)
    model.fit(X_train, y_train)
    model_metrics(model,X_test,y_test)




if __name__ == "__main__":


    t_artists=['Ivan Aivazovsky',
    'Marc Chagall',
    'Camille Pissarro',
    'Albrecht Durer',
    'Vincent Van Gogh',
    'Paul Cezanne',
    'Martiros Saryan',
    'Ivan Shishkin',
    'Gustave Dore',
    'Pierre-Auguste Renoir',
    'Rembrandt',
    'Pablo Picasso']


    #VGG16

    #If model_counter=1, then VGG16 model is built
    model_counter=1
    #Import VGG16 Features
    chunk = pd.read_csv("complete_info_extracted_features.csv",chunksize=50000)
    pd_df = pd.concat(chunk)

    #Subsetting the dataset by the 12 artists
    pd_dfartists=pd_df[pd_df['artist'].isin(t_artists)]

    #Contains the Artists label (Y in the model)
    df_art=list(pd_dfartists['artist'])


    X_traina, X_testa, y_traina, y_testa = train_test_split(pd_dfartists.iloc[:,1:25088], df_art, test_size=0.33, stratify=df_art)

    model(X_train, X_test, y_train, y_test)

    #ResNet50
    model_counter=0
    #Loading the Dataset
    #data_res_art contains the resnet features dataset for artist classification
    counter=0
    data_res_art=pd.DataFrame()
    for filename in glob.glob('ResNet50_Artist_2021_04_22_00_07_48/*.csv'):
        if counter==0:
            data_res_art=pd.read_csv(filename,header=None)
        else:
            df=pd.read_csv(filename,header=None)
            data_res_art=data_res_art.append(df)
        counter=counter+1

    X_traina1, X_testa1, y_traina1, y_testa1 = train_test_split(data_res_art.iloc[:,0:100351], data_res_art[100352], test_size=0.33, stratify=data_res_art[100352])
    model(X_traina1, X_testa1, y_traina1, y_testa1)
