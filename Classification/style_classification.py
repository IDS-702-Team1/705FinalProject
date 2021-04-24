import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import plotting_metrics


styles=['Northern Renaissance', 'Early Renaissance', 'Mannerism (Late Renaissance)' ,'High Renaissance','Baroque','Rococo',
 'Romanticism','Realism', 'Impressionism', 'Post-Impressionism','Art Nouveau (Modern)', 'Expressionism', 'Surrealism', 'Cubism','Analytical Cubism',
 'Synthetic Cubism', 'Abstract Art', 'Abstract Expressionism']


#Function is used to merge smaller styles into its parent style
def style_combination():
    pd_dfa['style'].mask(pd_dfa['style'] == 'Northern Renaissance', 'Renaissance', inplace=True)
    pd_dfa['style'].mask(pd_dfa['style'] == 'Mannerism (Late Renaissance)', 'Renaissance', inplace=True)
    pd_dfa['style'].mask(pd_dfa['style'] == 'Early Renaissance', 'Renaissance', inplace=True)
    pd_dfa['style'].mask(pd_dfa['style'] == 'High Renaissance', 'Renaissance', inplace=True)
    pd_dfa['style'].mask(pd_dfa['style'] == 'Synthetic Cubism', 'Cubism', inplace=True)
    pd_dfa['style'].mask(pd_dfa['style'] == 'Analytical Cubism', 'Cubism', inplace=True)
    pd_dfa['style'].mask(pd_dfa['style'] == 'Abstract Expressionism', 'Abstract Art', inplace=True)
    pd_dfa['style'].mask(pd_dfa['style'] == 'Rococo', 'Baroque', inplace=True)
    pd_dfa['style'].mask(pd_dfa['style'] == 'Post-Impressionism', 'Impressionism', inplace=True)
    return pd_dfa



def model_metrics(model,X_test,y_test,model_counter):

    # make predictions for test data
    y_pred_test = model.predict(X_test)
    #Model Classification Report
    cr_vgg_style=classification_report(y_test, y_pred_test))
    #Model Confusion Matrix
    cm_vgg_style=confusion_matrix(y_test,y_pred_test)

    #Plot the Confusion Matrix (Heat Map)
    if model_counter==1:

        title="Heat Map of VGG16 Painting Style Classification"
    else:
        title="Heat Map of ResNet50 Painting Style Classification"
    list_labels=['Abstract', 'Art Nouveau', 'Baroque',
                               ' Cubism', 'Expressionism', 'Impressionism','Realism', 'Renaissance',
                               'Romanticism', 'Surrealism']

    plotting_metrics.plot_confusion_matrix(cm_vgg_style,title,list_labels)

    #Plot the ROC Curve
    plotting_metrics.plot_roc(model,X_test,y_test)



def model(X_train, X_test, y_train, y_test,model_counter):

    model = XGBClassifier(n_estimators=250, n_jobs=-1)
    model.fit(X_train, y_train)
    model_metrics(model,X_test,y_test,model_counter)






if __name__ == "__main__":

    #VGG16
    #If model_counter=1, then VGG16 model is built
    model_counter=1
    #Import VGG16 Features


    chunk = pd.read_csv("complete_info_extracted_features.csv",chunksize=50000)
    pd_df = pd.concat(chunk)


    pd_dfa=pd_df[pd_df['style'].isin(styles)]

    pd_dfa=style_combination(pd_dfa)

    #datax contains the Image Features
    datax=pd_dfa.iloc[:,1:25088]

    #style_y contains the labels of the painting style
    style_y=list(pd_dfa.iloc[:,25099])

    X_train, X_test, y_train, y_test = train_test_split(datax, style_y, test_size=0.33, stratify=style_y)
    model(X_train, X_test, y_train, y_test,model_counter)

    #ResNet50
    #VGG=0 indicates that resnet is built
    model_counter=0
    #ResNet Data was in 4 folders, p1, p2, p3, p4.
    #combined_csv1 consists of the combination of all csv files in p1
    extension = 'csv'
    all_filenames1 = [i for i in glob.glob('p1/*.{}'.format(extension))]
    #combined_csv1 consists of the combination of all csv files in p1
    combined_csv1 = pd.concat([pd.read_csv(f,header=None) for f in all_filenames1 ])

    all_filenames2 = [i for i in glob.glob('p2/*.{}'.format(extension))]
    #combined_csv2 consists of the combination of all csv files in p2
    combined_csv2 = pd.concat([pd.read_csv(f,header=None) for f in all_filenames2 ])

    all_filenames3 = [i for i in glob.glob('p3/*.{}'.format(extension))]
    #combined_csv3 consists of the combination of all csv files in p3

    combined_csv3 = pd.concat([pd.read_csv(f,header=None) for f in all_filenames3 ])

    all_filenames4 = [i for i in glob.glob('p4/*.{}'.format(extension))]
    #combined_csv4 consists of the combination of all csv files in p4
    combined_csv4 = pd.concat([pd.read_csv(f,header=None) for f in all_filenames4 ])

    #final dataset for ResNet50
    final_res_data=pd.concat(combined_csv1,combined_csv2,combined_csv3,combined_csv4)
