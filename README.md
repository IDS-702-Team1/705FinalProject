# Using Hybrid CNN-XGBoost Classifier for Classification of Fine Art Paintings by Style and Artist

This project aims to classify fine-art paintings by their corresponding artists and styles. 


## Data

We obtained the data from a Kaggle Competition (Painters by Numbers: https://www.kaggle.com/c/painter-by-numbers). The dataset consists of over 100,000 paintings and has labeled each painting with its respective artist, genre, style and date of creation. 


## Repo Documentation:
1) The folder **Data** consists of the 2 links that redirects you to the AWS S3 bucket. We stored the features extracted from VGG16 and ResNet50 in this bucket. We have stored the data on Google Drive as well. 

2) The folder **FeatureExtracting** consists of the code to extract features from the images via CNNs. The main code is in `create_image_feature.py`. You can find a series of the function named "gen_*model_name*()" at the bottom of the file. This function can be used directly to generate image features. The meaning of parameter:

  - Output: The folder used to receive generated features. The function would automatically generate a subfolder with timestamps under the output. 
  - image_size: Input size of CNN module. The default size is $224 \times 224$. Images that do not follow the size will be resized automatically. 
  - batch_size: This parameter represents how many image features will be grouped in a small batch file. The size is dependent on the module and computer memory. Recommend value is 50 ~ 200. 
  - Job: This parameter shows how many jobs would work in parallel. It should be not larger than the CPU core number. If the value is identical to the CPU core number means the computer is, this task will occupy all computational resources. 

4) The folder **Classification** consists of the main py script that is used to classify paintings into Styles and Artists using XGBoost. The Python Script `artist_classification.py` will load the ResNet50 and VGG16 feature data for artists, split the data into training and testing, train the model using XGBoost, and then run the predictions on the testing data. `style_classification.py` will do similar functionalities as `artist_classification.py` but for the styles data. The Python script `plotting_metrics.py` is used to plot the Confusion Matrix and the ROC Curves.

5) The folder **Data Preprocessing** walks through the Data Preprocessing steps we computed for the project.
6) The folder **notebook** contains rough working of the model training and testing. (It is not the main code in the repo. Just for Reference for the team).


### How to Run the Repo
Download all data from S3 in the same repository.
(Including the Images for feature extraction)
1) Feature Extraction
```python
python create_image_feature.py
```
2) Artist Classification

```python
python artist_classification.py
```
3) Style Classification
```python
python style_classification.py
```
## Model Architecture

Upon reading research papers, we decided to implement a Hybrid CNN-XGBoost Model where the CNN model would extract the painting features and the XGBoost Classifier would classify the paintings into the respective artist/style. Research Papers indicated that this hybrid model is 1) Computationally less expensive and 2) Produce similar/even better results than the original CNN model and hence we decided to use this model. 

We decided to use two different CNN networks, namely the VGG-16 and ResNet-50 to extract image features and XGBoost for classification. The architectures are shown below.


![Screen Shot 2021-04-24 at 2 47 34 PM](https://user-images.githubusercontent.com/30974949/115969656-02cb4a80-a50c-11eb-8fc4-37770e8d1ddf.png)
<div align="center">
Figure 1. Architecture of Hybrid VGG16-XGBoost Model
</div>

![VGG16](https://github.com/IDS-702-Team1/705FinalProject/blob/main/reports/img/ResNet50.png)
<div align="center">
Figure 2. Architecture of Hybrid VGG16-XGBoost Model
</div>

### Style Classification

The styles we chose for this classification model ranged from the 1400s to 2000s. They were:

| Painting Style  | Time Period |
| :---            | :---:       |
| Renaissance     | 1400-1600   |
| Baroque         | 1600-1750   |
| Romanticism     | 1800-1850   |
| Realism         | 1850-1860   |
| Impressionism   | 1860-1870   |
| Art Nouveau     | 1880-1910   |
| Expressionalism | 1905-1920   |
| Surrealism      | 1910-1920   |
| Cubism          | 1900-1920   |
| Abstract Art    | 1940+  |


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
