# 705FinalProject

This project aims to identify paintings by their respective artists and styles. 


### Data

We obtained the data from a Kaggle Competition (Painters by Numbers). The dataset consisted of over 100,000 paintings and was classified by the respective artist, genre, style and time period. 


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

### Model Architecture

Upon reading research papers, we decided to implement a Hybrid CNN-XGBoost Model where the CNN model would extract the painting features and the XGBoost Classifier would classify the paintings into the respective artist/style. Research Papers indicated that this hybrid model is 1) Computationally less expensive and 2) Produce similar/even better results than the original CNN model and hence we decided to use this model. 

We decided to use two different CNN networks, namely the VGG-16 and ResNet-50 to extract image features. ResNet-50 is a convolutional neural network that is 50 layers deep while VGG-16 is a convolutional neural network that is 16 layers deep. 

<Insert Model Architecture Diagram Here>
  

###
