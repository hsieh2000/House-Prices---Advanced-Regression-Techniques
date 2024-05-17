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

There is worth noting detail that the validation set will be standardized by the mean and std of the training set.  
According to online resource and Chat-GPT, I conclude several point why we have to do that:

1. **Consistency**: During model training, the model learns from the training data, which includes the distribution of features. By standardizing the validation set using the mean and standard deviation of the training set, you are ensuring that the validation data is transformed in the same way as the training data. This consistency is important for fair evaluation of the model's performance.

2. **Avoiding Data Leakage**: Standardizing the validation set using statistics computed from the training set helps to avoid data leakage. Data leakage occurs when information from the validation set unintentionally influences the model training process, leading to overly optimistic performance estimates. By standardizing the validation set separately from the training set, you prevent this leakage and obtain more reliable performance estimates.

3. **Real-world Generalization**: In real-world scenarios, the model will encounter new data with potentially different distributions than the training data. By standardizing the validation set using the training set statistics, you are simulating this scenario and evaluating how well the model generalizes to new, unseen data.

# Construct Network  
Before I apply neuron network to solve this regression problem, I uesd polynomial regression with back elimination to predict SalePrice as the baseline, which get 0.17353 on public score. The score is probably in the bottom 30% of the leaderboard.  

When I firstly construct the network, the architecture is really simple, constructed by two Dense layers and one ReLU activation function, the loss function is MSELoss, and the optimizer is Adam.  The public score of this primitive model prediction is 0.16941, which already transcend the baseline, the loss of validation set is pretty high compares to the final model though. 

Since the validation loss still high, I do lots of attempts to diminish it.  
1. **Netowork Architecture**: I increase complexity of model by adding multiple layers, the performance of model does improve. However, the improvement seems limited. And I find out the when the adding layers exceeds a certain number, both training and validation loss will encounter a severe oscillation and eventually end with higher loss value then the model prediction.

2. **Loss Function**: I observe the training dataset and found out that there are several outliers, I think maybe these outliers undermine performance of the model, so I apply different loss function, such as L1Loss, SmoothL1Loss, and HuberLoss, to see if the loss value will decrease or not.  And the loss value actually comes low, but it is still high. After look up the formula of each loss function, I understand that when loss value is small, MSE and MAE have similiar outcome; Nevertheless, when loss value exceeds a certain range, MSE will increase exponentially, on the other hand, MAE will keep increase linearly. So the change of loss function doesn't actually ameliorate the performance, the ground turth and the prediction is still large, it just appears smaller when I choose MAE kind loss functions.
![image]("pic/截圖 2024-05-17 下午5.12.32.png")

4. **Optimizer**: Since changing netowork architecture doesn't have significant improvement and even cause the worse result, I try to cut through from different angle. Theoretically, the deeper netowork can do and shoud do the work better than the naive netowork, if not, I think I probably have an optimization issue. I switch the optimizer from Adam to the other often-used optimizer - SGD, and set learning rate to 0.01, momentum to 0.9(based on exprience, momentum = 0.9 normally get the best performance). But the result only changes slightly , it's too small to let me tell if the performance is actually improved or just a random coincidence. So I switch the optimizer back to Adam, but modify the learning rate to 0.001, at that time, the loss function is HuberLoss and its hyperparameter *delta* is 0.1(which is non-reasonable, since *delta* is the threshold of `|y - ŷ|`, , the scale of label about five-digit to six-digit and the model performance is not accurate, it is impossible to have the *delta* < 0.1) 
5.  I really shock about the change of loss value
6. 









