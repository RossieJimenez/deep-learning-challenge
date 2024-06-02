# Alphabet Soup Charity Funding Predictor

## Overview
The Alphabet Soup Charity Funding Predictor is a machine learning project aimed at helping Alphabet Soup, a nonprofit foundation, identify applicants for funding who are most likely to succeed. By leveraging historical data on previous applicants and their success rates, we build a binary classification model using deep learning. This model predicts the likelihood of success for new applicants based on various features such as application type, affiliation, classification, use case for funding, organization type, income classification, and other relevant metadata.

### Project Description
Alphabet Soup aims to allocate its funding resources more effectively by identifying applicants who are most likely to succeed. This project uses historical data of over 34,000 organizations to train a binary classification model. The goal is to maximize the accuracy of predicting whether an applicant will be successful if funded.

### Data Preprocessing
The dataset includes several features and a target variable IS_SUCCESSFUL indicating whether the funding was used effectively. The preprocessing steps include:

* Dropping the non-beneficial ID column EIN.
* Encoding categorical variables using pd.get_dummies.
* Scaling the data using StandardScaler.
* Splitting the data into training and testing sets.
* Replacing rare values in the NAME column with "Other" (This is for the Optimization process).

### Model Architecture and Training
We used a neural network model implemented with TensorFlow and Keras. The model's architecture and the training process included the following steps for pre optimization:

* Dropped EIN and NAME columns.
* Initial model with 2 hidden layers.
    * Included two hidden layers with 8 and 3 neurons respectively.
    * Achieved an accuracy of 72.60%.
   
## Optimization Attempts: 1
### First Optimization
* Changes: Dropped the EIN column and added NAME column back, set the test size to 20%, and included three hidden layers with 33, 28, and 18 neurons respectively.
* Accuracy: 78.57%
* Observations: Adding the NAME column proved to be a good choice compared to the unoptimized version. This shows the importance of choosing the right features for the model.

### Results
The final model achieved an accuracy of 78%, exceeding the target accuracy of 75%. The key to this improvement was the reintroduction and preprocessing of the NAME column, which provided additional valuable information to the model.

### Future Work
To further validate and potentially improve the model, we recommend exploring other models such as the Random Forest Classifier. Random Forests are robust to overfitting and can handle categorical data effectively without extensive preprocessing. They also provide feature importance, which can help in understanding the most predictive features.
