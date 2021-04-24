import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

def plot_confusion_matrix(cf_matrix,title,list_labels):

    #Confusion Matrix values are the percentage of datapoints that belongs to a cell for each row
    #The row values add up to 1
    for i in range(np.shape(np.array(cf_matrix))[0]):
        sumcol=sum(cf_matrix[i][:])
        for j in range(np.shape(np.array(cf_matrix))[1]):
            cf_matrix[i][j]=round(cf_matrix[i][j]/sumcol,2)


    fig, ax = plt.subplots(figsize=(20,20))
    im = ax.imshow(cf_matrix)
    #Showing all ticks
    ax.set_xticks(np.arange(len(cf_matrix)))
    ax.set_yticks(np.arange(len(cf_matrix))) # Labelling ticks ax.set_xticklabels(no_nodes) ax.set_yticklabels(no_layers) ax.set_ylim(-1, 4)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),  ha="right",size=60,
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right",size=60)
    # Loop over data dimensions and create text annotations.
    ax.set_ylim(16,-8)


    for i in range(np.shape(np.array(cf_matrix))[0]):
        if i==0:
            continue
        else:
            for j in range(np.shape(np.array(cf_matrix))[1]):
                text = ax.text(j, i, round(cf_matrix[i][j],2),color='r',size=10)


    ax.set_title("Heat Map of VGG16 Painting Style Classification ",size=20)
    fig.tight_layout()
    plt.xlabel("Actual Style",size=20)
    plt.ylabel("Predicted Style",size=10)

    plt.xticks(np.arange(len(list_labels)), list_labels,
               fontsize=10,va='center')
    plt.yticks(np.arange(len(list_labels)), list_labels,
               fontsize=10)
    plt.legend()
    plt.show()


def plot_roc(model,X_test,y_test):
    y_predproba_test = model.predict_proba(X_test)

    #Encoding the Y_test
    enc = OneHotEncoder(handle_unknown='ignore')
    label_encoder = LabelEncoder()
    #Integer encoding the styles
    integer_encoded = label_encoder.fit_transform(y_test)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    #One hot encoded the y_test labels
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    #nld is a dictionary that contains the integer encoded as key and style name as value
    nld={}
    for n,l in zip(integer_encoded,y_test):
        n=n[0]
        if n not in nld.keys():
            nld[n]=l

    # Compute ROC curve and ROC area for each class
    #False Postive Rate
    fpr = dict()
    #True Positive Rate
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve( onehot_encoded[:, i],y_predproba_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(10):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 10

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    lw = 2

    # Plot all ROC curves
    plt.figure(figsize=(20,10))


    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','blue','green','purple','pink','brown','yellow'])
    for i, color in zip(range(10), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f})'
                 ''.format(nld[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Styles of Paintings')
    plt.legend(loc="lower right")
    plt.show()
