# 705FinalProject

This project aims to identify paintings by their respective artists and styles. 


### Data

We obtained the data from a Kaggle Competition (Painters by Numbers: https://www.kaggle.com/c/painter-by-numbers). The dataset consisted of over 100,000 paintings and was classified by the respective artist, genre, style and time period. 


### How to Use the Repo:
1) The Folder **Data** consists of the 2 links that redirects you to the AWS s3 bucket. We stored the VGG16 and ResNet50 features in this bucket. We have stored the data on Google Drive as well. 
2) The Folder **Feature Extraction** consists of the code to extract features from the images via Resnet and VGG16. 



4) The Folder **Classification** consists of the main py script that is used to classify paintings into Styles and Artists using XGBoost. The Python Script "artist_classification.py" will load the ResNet50 and VGG16 feature data for artists, split the data into training and testing, train the model using XGBoost, and then run the predictions on the testing data. The "style_classification.py" will do similar functionalities as "artist_classification.py" but for the styles data. The Python script "plotting_metrics.py" is used to plot the Confusion Matrix and the ROC Curves.
5) 



### Model Architecture

Upon reading research papers, we decided to implement a Hybrid CNN-XGBoost Model where the CNN model would extract the painting features and the XGBoost Classifier would classify the paintings into the respective artist/style. Research Papers indicated that this hybrid model is 1) Computationally less expensive and 2) Produce similar/even better results than the original CNN model and hence we decided to use this model. 

We decided to use two different CNN networks, namely the VGG-16 and ResNet-50 to extract image features.
VGG16 Model Architecture:

![Screen Shot 2021-04-24 at 2 47 34 PM](https://user-images.githubusercontent.com/30974949/115969656-02cb4a80-a50c-11eb-8fc4-37770e8d1ddf.png)

Resnet50 Model Architecture:

![Screen Shot 2021-04-24 at 2 48 43 PM](https://user-images.githubusercontent.com/30974949/115969686-2bebdb00-a50c-11eb-816d-80c472057601.png)
###


### Style Classification

The styles we chose for this classification model ranged from the 1400s to 2000s. They were:

1. Renaissance (1400-1600)
2. Baroque (1600-1750)
3. Romanticism (1800-1850)
4. Realism (1850-1860)
5. Impressionism (1860-1870)
6. Art Nouveau - Modern (1880-1910)
7. Expressionalism (1905-1920)
8. Surrealism (1910-1920)
9. Cubism (1900-1920)
10. Abstract Art (1940+)


### Artist Classification

The artists we chose for this classification model were:
1. Ivan Aivazovsky
2. Marc Chagall
3. Camille Pissarro
4. Albrecht Durer
5. Vincent Van Gogh
6. Paul Cezanne
7. Martiros Saryan
8. Ivan Shishkin
9. Gustave Dore
10. Pierre-Auguste Renoir
11. Rembrandt
12. Pablo Picasso

We chose these artists as they had a minimum of 500 paintings and represented the 10 styles above.
