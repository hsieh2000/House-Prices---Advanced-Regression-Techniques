# House-Prices---Advanced-Regression-Techniques
KAGGLE · GETTING STARTED PREDICTION COMPETITION 

This is a self-learning project.  
Trough this project, I got much more familiar with neuron network, and realize how to adjust hyperparameters and optimize the performance of the model.

The dataset can be downloaded [here](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/)

# Preprocessing
Firstly, I observe train and test dataset to make sure that two datasets follow the same distrbution.

Next, I split data into two types - numerical and categroical data,
and the categroical data could be futher divided into ordinal and non-ordinal data.

To encode categroical data, I load *data_description.txt* and save the categroies of each feature.
