# House-Prices---Advanced-Regression-Techniques
KAGGLE · GETTING STARTED PREDICTION COMPETITION 

This is a self-learning project.  
Trough this project, I got much more familiar with neuron network, and realize how to adjust hyperparameters and optimize the performance of the model.

The dataset can be downloaded [here](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/)

# Preprocessing
Firstly, I observe the train and the test dataset to make sure that two datasets follow the same distrbution.

Next, through observing the dtype of dataset, I split data into two types - numerical and categroical data,
and the categroical data could be futher divided into ordinal and non-ordinal data.

To encode categroical data, I load *data_description.txt* and extract the information of each feature's categroies.

For ordinal categroical data, I apply `sklearn.preprocessing.OrdinalEncoder` to encode data, and transfer nan value with -1.

For non-ordinal categorical data, I use `sklearn.preprocessing.OneHotEncoder`. However, I have found that in certain features, the names of categories in *data_description.txt* do not match those in the dataset. To resolve this discrepancy, I adjust the category names in the dataset according to *data_description.txt*.

