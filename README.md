# House-Prices---Advanced-Regression-Techniques
KAGGLE · GETTING STARTED PREDICTION COMPETITION 

This is a self-learning project.  
Trough this project, I got much more familiar with neuron network, and realize how to adjust hyperparameters and optimize the performance of the model.

The dataset can be downloaded [here](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/)

# Preprocessing
Firstly, I observe the train and the test dataset to make sure that two datasets follow the same distrbution.

Next, by observing the data types of the dataset, I split the data into two types: numerical and categorical data. The categorical data can be further divided into ordinal and non-ordinal data.

To encode categroical data, I load *data_description.txt* and extract the information of each feature's categroies.

For ordinal categorical data, I use `sklearn.preprocessing.OrdinalEncoder` to encode the data and replace missing values with -1.

For non-ordinal categorical data, I use `sklearn.preprocessing.OneHotEncoder`. However, I have found that in certain features, the names of categories in *data_description.txt* do not match those in the dataset. To resolve this discrepancy, I adjust the category names in the dataset according to *data_description.txt*.

For numerical data, I fill missing values with the mean of each feature to prevent errors when inputting data into the neural network model.

# Create Dataset
I have created a class called `HOUSEDataset` to handle the dataset. I transfer data from NumPy arrays to tensors and standardize numerical features. This process helps increase the convergence of the model and assists in preventing issues such as exploding gradients and vanishing gradients.








