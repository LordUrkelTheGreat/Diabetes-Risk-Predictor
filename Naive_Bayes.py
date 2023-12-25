# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import math
from sklearn.metrics import mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# pima function
def pima():
    # load in the dataset
    pima_dataset = pd.read_csv('diabetes.csv')
    
    # print the first 5 rows of the dataset
    print("Pima Dataset Head:\n", pima_dataset.head(), "\n")
    
    # copy the dataset
    copy_pima = pima_dataset.copy(deep = True)
    
    # the columns that have invalid values are: Glucose, BloodPressure, SkinThickness, Insulin, and BMI
    # replace the invalid valid values in these columns with NaN values
    copy_pima[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = copy_pima[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)
    
    # print the number of NaN values in each column
    print("Number of NaN values:")
    print(copy_pima.isnull().sum(), "\n")
    
    # plot the dataset
    pima_plot = pima_dataset.hist(figsize = (10, 10))
    
    # replace the NaN values with the mean values of their respective columns
    copy_pima["Glucose"].fillna(copy_pima["Glucose"].mean(), inplace = True)
    copy_pima["BloodPressure"].fillna(copy_pima["BloodPressure"].mean(), inplace = True)
    copy_pima["SkinThickness"].fillna(copy_pima["SkinThickness"].mean(), inplace = True)
    copy_pima["Insulin"].fillna(copy_pima["Insulin"].mean(), inplace = True)
    copy_pima["BMI"].fillna(copy_pima["BMI"].mean(), inplace = True)
    
    # plot the copied dataset
    copy_plot = copy_pima.hist(figsize = (10, 10))
    
    # print the copied dataset
    print("Copied Dataset:\n", copy_pima, "\n")
    
    # store the data into X and y
    X = copy_pima.iloc[:, 0:8].values       # take features 0 through 8
    y = copy_pima["Outcome"].values         # take the outcome column
    
    # return X and y
    return X, y

# early function
def early():
    # load in the dataset
    early_dataset = pd.read_csv('diabetes_data.csv', sep=';')
    
    # print the first 5 rows of the dataset
    print("Early Dataset Head:\n", early_dataset.head(), "\n")
    
    # print the number of NaN values in each column
    print("Number of NaN values:")
    print(early_dataset.isnull().sum(), "\n")
    
    # plot the dataset
    early_plot = early_dataset.hist(figsize = (10, 10))
    
    # change the gender column to numerical values
    label_encoder = LabelEncoder() 
    early_dataset['gender'] = label_encoder.fit_transform(early_dataset['gender'])
    
    # store the data into X and y
    X = early_dataset.iloc[:, 0:15].values      # take features 0 through 16
    y = early_dataset.iloc[:, 16].values       # take the last column
    
    # plot the dataset
    early_plot = early_dataset.hist(figsize = (15, 15))
    
    return X, y

# split and standardize the dataset
def split_and_std(X, y, train_index, test_index):
    # split the dataset into training and testing data using the indices
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    
    # standardize the training and testing data
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    
    # return the splitted and standardized data
    return X_train, X_test, y_train, y_test

# one hot encoding
def one_hot_encoding(y):
    onehot_encoder = OneHotEncoder(sparse = False)
    integer_encoded = y.reshape(len(y), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

# arg max function
def arg_max(array):
    # find the largest value
    max_value = max(array)
    
    # loop through every array element
    for i in range(len(array)):
        # store 1 at array element if it is greater than or equal to the max
        # store 0 at array element if it is less than the max
        if (array[i] >= max_value):
            array[i] = 1.0
        else:
            array[i] = 0
    
    # return the array
    return array

# gauss probability
def gauss_probability(mean, std, x):
    # the 1e-300 is to prevent dividing by 0
    return np.exp((-np.power((x - mean), 2)) / (2 * (np.power(std, 2)) + (1e-300)))

# find the mean and standard deviation for each class
def mu_std(X, y, features, classifications):
    # create the mean and standard deviation matrices
    mean = np.zeros((features, classifications), float)
    std = np.zeros((features, classifications), float)
    
    # loop through the classifications
    for c in range(classifications):
        # loop through the features
        for f in range(features):
            # find the y values that match the current classification
            match = np.where(y == c)
            
            # calculate the mean and standard deviation of the current feature and classification
            mean[f][c] = X[match, f].mean()
            std[f][c] = X[match, f].std()
            
    # return the mean and standard deviation
    return mean, std

# naive bayes algorithm
def naive_bayes(X, y, mean, std, features, classifications, observations):
    # create a prediction matrix
    P = np.ones((observations, classifications), float)
    
    # copy the prediction matrix
    P_argmax = P
    
    # loop through the observations
    for (i, x) in enumerate(X):
        # loop through the classifications
        for c in range(classifications):
            # calculate P(c)
            Pc = y.tolist().count(c) / observations
            
            # loop through the features
            for f in range(features):
                # calculate the probability of the current feature, classification, and sample
                P[i][c] = P[i][c] * gauss_probability(mean[f][c], std[f][c], x[f])
                
            # multiply the probability from the previous calculation
            P[i][c] = P[i][c] * Pc
            
        # get the argmax at the current sample
        P_argmax[i] = arg_max(P[i])
        
    # return the argmax prediction matrix
    return P_argmax

# change the prediction argmax back to the original classifications
def prediction_classification(y, P_argmax, classifications):
    # create a target prediction matrix
    y_pred = np.zeros((len(y)), int)
    
    # loop through the predictions
    for (i, x) in enumerate(P_argmax):
        # flag to prevent overwrite
        flag = False
        
        # loop through the classifications
        for c in range(classifications):
            # check if the prediction at the current classification is equal to 1
            # also check if the flag is not set to True
            if ((P_argmax[i][c] == 1) and (flag != True)):
                # add the current classification to the list
                y_pred[i] = c
                
                # set the flag to True
                flag = True
    
    # return the target prediction matrix
    return y_pred

# find the total number of misclassified samples
def misclassified_samples(y_pred, y):
    # variable that stores the total number of misclassified samples
    m_s = 0
    
    # loop through the prediction and target matrices
    for i in range(len(y)):
        # check if the prediction is not equal to the target
        if (y_pred[i] != y[i]):
            # iterate the misclassified samples
            m_s = m_s + 1
            
    # return the total number of misclassified samples
    return m_s

# calculate the accuracy
def accuracy(y_pred, y):
    # calculate the accuracy between the prediction and target
    a = accuracy_score(y, y_pred)
    
    # calculate the root squared mean error
    MSE = np.square(np.subtract(y, y_pred)).mean()   
    rsme = math.sqrt(MSE)  
    
    # calculate the mean absolute percentage error
    mape = mean_absolute_error(y, y_pred) * 100
    
    # calculate the fscore, precision, and recall
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    fscore = f1_score(y, y_pred)
    
    # return the accuracy scores
    return a, rsme, mape, precision, recall, fscore

# find the misclassified indices
def error_indices(y_pred, y):
    # where were the errors located at
    err = np.where(y != y_pred)
    
    # return the error indices
    return err

# run algorithm
def algorithm(X, y, string, flag, gnb):
    # print divider
    #print("=======================================================================================================================================")
    
    # print information
    #print(string, "DATA:")
    
    # print divider
    #print("=======================================================================================================================================")
    
    # turn the predicted matrix into a one hot coded matrix
    onehot_y = one_hot_encoding(y)
    
    # get the number of classifications, features, and observations
    classifications = len(onehot_y[0])
    features = len(X[0])
    observations = len(X[:, 0])
    
    # find the mean and standard deviation for each feature and class
    mean, std = mu_std(X, y, features, classifications)
    
    # do naive bayes
    P_argmax = naive_bayes(X, y, mean, std, features, classifications, observations)
    
    # calculate the prediction
    if (flag == False):
        y_pred = prediction_classification(y, P_argmax, classifications)
    elif (flag == True):
        y_pred = gnb.predict(X)
        
    # find the total number of misclassified samples
    misclassified_sample = misclassified_samples(y_pred, y)
    
    # calculate the accuracy
    accurate, rsme, mape, precision, recall, fscore = accuracy(y_pred, y)
    
    # find out where the errors occurred
    err = error_indices(y_pred, y)
    
    # print the information
    #print("Number of", string, "Samples: ", observations)
    #print("Number of", string, "Features: ", features)
    #print("Number of", string, "Misclassified Samples: ", misclassified_sample)
    #print(string, "Accuracy: ", accurate)
    #print(string, "RSME: ", rsme)
    #print(string, "MAPE: ", mape)
    #print("Errors at Indices: ", err)
    #print("Actual Classification: ", y[err])
    #print("Predicted Classification: ", y_pred[err])
    
    # print divider
    #print("=======================================================================================================================================\n")
    
    # return the number of misclassified samples and the accuracy score
    return misclassified_sample, accurate, precision, recall, fscore

# k fold algorithm
def k_fold_algorithm(k_folds, X, y, string, flag):
    # create lists to store the accuracy/misclassified samples per fold
    training_accuracy = []
    testing_accuracy = []
    training_misclassified_samples = []
    testing_misclassified_samples = []
    
    # create lists to store the fscore, precision, and recall scores
    training_fscore = []
    testing_fscore = []
    training_precision = []
    testing_precision = []
    training_recall = []
    testing_recall = []
    
    # loop through the folds
    for (training_index, testing_index) in k_folds.split(X, y):
        # split and standardize the data
        X_train, X_test, y_train, y_test = split_and_std(X, y, training_index, testing_index)
        
        # create sklearn naive bayes
        gnb = GaussianNB()
        
        # fit the training data
        gnb.fit(X_train, y_train)
        
        # pass training data through the algorithm
        m_s, a, p, r, fs = algorithm(X_train, y_train, string, flag, gnb)
        
        # add the accuracy score to the training list
        training_accuracy.append(a)
        
        # add the misclassified samples to the training list
        training_misclassified_samples.append(m_s)
        
        # add the precision to the training list
        training_precision.append(p)
        
        # add the recall to the training list
        training_recall.append(r)
        
        # add the fscore to the training list
        training_fscore.append(fs)
        
        # pass testing data through the algorithm
        m_s, a, p, r, fs = algorithm(X_test, y_test, string, flag, gnb)
        
        # add the accuracy score to the testing list
        testing_accuracy.append(a)
        
        # add the misclassified samples to the testing list
        testing_misclassified_samples.append(m_s)
        
        # add the precision to the testing list
        testing_precision.append(p)
        
        # add the recall to the testing list
        testing_recall.append(r)
        
        # add the fscore to the testing list
        testing_fscore.append(fs)
        
    # print the information
    print("Average Number of", string, "Training Misclassified Samples: ", math.ceil(sum(training_misclassified_samples) / len(training_misclassified_samples)))
    print("Average", string, "Training Accuracy: ", sum(training_accuracy) / len(training_accuracy))
    print("Average", string, "Training Precision: ", sum(training_precision) / len(training_precision))
    print("Average", string, "Training Recall: ", sum(training_recall) / len(training_recall))
    print("Average", string, "Training F1 Score: ", sum(training_fscore) / len(training_fscore))
    print("Average Number of", string, "Testing Misclassified Samples: ", math.ceil(sum(testing_misclassified_samples) / len(testing_misclassified_samples)))
    print("Average", string, "Testing Accuracy: ", sum(testing_accuracy) / len(testing_accuracy))
    print("Average", string, "Testing Precision: ", sum(testing_precision) / len(testing_precision))
    print("Average", string, "Testing Recall: ", sum(testing_recall) / len(testing_recall))
    print("Average", string, "Testing F1 Score: ", sum(testing_fscore) / len(testing_fscore))
    print("\n")
    
    return

# main function
def main():
    # create k folds
    k_folds = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
    
    # print divider
    print("=======================================================================================================================================")
    
    # print information
    print("PIMA DATASET: ")
    
    # print divider
    print("=======================================================================================================================================\n")
    
    # get the X and y matrices from the dataset
    X, y = pima()
    
    # pass the data through the k folds algorithm
    k_fold_algorithm(k_folds, X, y, "PIMA", False)
    
    # pass the sklearn data through the k folds algorithm
    k_fold_algorithm(k_folds, X, y, "SKLEARN PIMA", True)
    
    # print divider
    print("=======================================================================================================================================")
    
    
    
    
    
    
    # print divider
    print("\n\n\n\n\n=======================================================================================================================================")
    
    # print information
    print("EARLY DATASET: ")
    
    # print divider
    print("=======================================================================================================================================\n")
    
    # get the X and y matrices from the dataset
    X, y = early()
    
    # pass the data through the k folds algorithm
    k_fold_algorithm(k_folds, X, y, "EARLY", False)
    
    # pass the sklearn data through the k folds algorithm
    k_fold_algorithm(k_folds, X, y, "SKLEARN EARLY", True)
    
    # print divider
    print("=======================================================================================================================================")
    
    return

# call main
main()