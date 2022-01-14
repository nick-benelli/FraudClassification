#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:49:21 2021

@author: nickbenelli
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.metrics
from sklearn.metrics import confusion_matrix


def map_string_values_ints(df, string_colnames = []):
    '''
    Converts strings to ints in data frame

    Parameters
    ----------
    df : pandans.DataFrame
        data frame with string values.
    string_colnames : list, optional
        List of columns with string variables. The default is [].

    Returns
    -------
    df : pandans.DataFrame
        data frame with all int values.
    map_dict : dict
         map dictionary of strings converted to ints.

    '''    
    map_dict = {}
    non_number_starter = 99
    
    for colname in string_colnames:
        # create key fo colname
        map_dict[colname] = {}
        non_number_idx = non_number_starter
        
        # Loop through - convert string to integers
        for item in set(df.loc[:, colname]):
            try:
                map_dict[colname][item] = int(item)
            except ValueError:
                pass

        # Loop through values that could not convert string to int and assign number 99  
        for item in set(df.loc[:, colname]):
            if item not in map_dict[colname].keys():
                while non_number_idx in map_dict[colname].values():
                    non_number_idx -= 1
                
                map_dict[colname][item] = non_number_idx
        df.loc[:, colname] = df.loc[:, colname].map(map_dict[colname])
    print(map_dict)
    return df, map_dict


def fill_nas_df(df, use_mean= True):
    '''
    Replaces NAs in data frame column with mean or median.

    Parameters
    ----------
    df : pandas.DataFrame
        data frame with NAs in float.
    use_mean : bool, optional
        True- Mean
        False- Median. The default is True.

    Returns
    -------
    df : pandas.DataFrame
        data frame with filled in NAs.

    '''
    # Fill Blanks
    for col in df.columns:
        try:
            if use_mean:
                replace_value = df.loc[:, col].mean()
            else:
                 replace_value = df.loc[:, col].median()
            df.loc[:, col].fillna(value=replace_value, inplace=True)
        except TypeError:
            continue
    return df


def convert_int_to_float_df(df):
    '''
    Converts int to floats

    Parameters
    ----------
    df : pandas.DataFrame
        df input.

    Returns
    -------
    df : pandas.DataFrame
        df output.

    '''
    for col in df.columns:
        try:
            #df.loc[:, col] = df.loc[:, col].astype(float)
            df.loc[:, col] = pd.to_numeric(df.loc[:, col], downcast='float')
        except ValueError:
            pass
    return df


def dummy_string_var(df, string_colnames):
    '''
    Takes String Values and adds dummy data in place of column

    Parameters
    ----------
    df : pandas
        data frame.
    string_colnames : list
        list of string columns in data frame.

    Returns
    -------
    df_new : pandas
        data frame with new dummy variables.

    '''
    df_new = df.copy()
    col = string_colnames[0]
    dummy_dict = {}
    for col in string_colnames:
        df_sting = df_new.loc[:, col]
        df_dummy = pd.get_dummies(df_sting)
        
        # Rename the columns
        dummy_colname = list(df_dummy.columns)
        for inc, dummy_name in enumerate(dummy_colname):
            dummy_colname[inc] = '_'.join([col, dummy_name])
            
        df_dummy.columns = dummy_colname
        
        # Add dummy df to dict
        dummy_dict[col] = df_dummy
        # Add dummy data to dataframe and remove string col
        col_index_place = df_new.columns.get_loc(col)
        df_before_dummy = df_new.iloc[:, :col_index_place]
        df_after_dummy = df_new.iloc[:, (col_index_place+1):]
        df_new =  pd.concat([df_before_dummy, df_dummy, df_after_dummy], axis=1)
    
    return df_new

'''
# Accuracy
'''
def accuracy_table(y_test, y_predicted, test_name= ''):
    '''
    Builds confusion matrix and determines accuracy of algorithm

    Parameters
    ----------
    y_test : pd.Series or np.darray
        actual values in dataset.
    y_predicted : pd.Series or np.darray
        predicted values of dataset.
    test_name : str, optional
        Name of test run to predict. The default is ''.

    Returns
    -------
    accuracy : float
        accuracy score from sklearn metrics.
    conf_mat : numpy.darray
        confusion matrix from sklearn matrix.

    '''
    
    class_report = sklearn.metrics.classification_report(y_test, y_predicted)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_predicted)
    print("{} Classification Report: \n\n".format(test_name), class_report)
    print("{} Accuracy: {}".format(test_name, round(accuracy, 4)))
    print()
    conf_mat = confusion_matrix(y_test, y_predicted)
    #print('Confusion Matrix:\n{}'.format(conf_mat))
    
    # Plot confusion matrix
    #sns.set(font_scale = 0.8)
    sns.heatmap(conf_mat, square=True, annot=True, cbar=False, fmt='g')
    plt.xlabel('predicted value')
    plt.ylabel('true value')
    
    # Create Dictionary
    TN, FP, FN, TP = conf_mat.flatten()
    round_digit = 4
    precision = round(sklearn.metrics.precision_score(y_test, y_predicted), round_digit)
    recall = round(sklearn.metrics.recall_score(y_test, y_predicted), round_digit)
    f1_sc = round(sklearn.metrics.f1_score(y_test, y_predicted), round_digit)
    accuracy_dict = {
        'Test Name' : test_name, 
        'TP' : TP, 'TN' : TN, 'FP': FP, 'FN' : FN, 
        'Accuracy' : round(accuracy, round_digit), 
        'Precision' : precision, 'Recall' : recall, 'F1 Score' : f1_sc,
        'Confusion Matrix' : conf_mat}
    
    return accuracy_dict


def confusion_matrix_and_tests(y, y_predicted):
    '''
    Uses confusion matrix to find specicifity and sensitivy of model

    Parameters
    ----------
    y : pd.Series or np.darray
        actual output values of dataset.
    y_predicted : pd.Series or np.darray
        predicted output values of dataset.

    Returns
    -------
    conf_matrix_dict : dict
        dictionary of the specificity andn sensitivity.

    '''
    conf_matrix = confusion_matrix(y, y_predicted)
    conf_matrix_dict = {
        'TN' : conf_matrix[0, 0],
        'FN' : conf_matrix[1, 0],
        'TP' : conf_matrix[1, 1],
        'FP' : conf_matrix[0, 1],
        }
    
    conf_matrix_dict['sensitivity'] = conf_matrix_dict['TP'] / (conf_matrix_dict['TP'] + conf_matrix_dict['FN'])
    conf_matrix_dict['specificity'] = conf_matrix_dict['TN'] / (conf_matrix_dict['TN'] + conf_matrix_dict['FP'])
    
    conf_matrix_dict['sensitivity_percent'] = round(conf_matrix_dict['sensitivity'] * 100, 2)
    conf_matrix_dict['specificity_percent'] = round(conf_matrix_dict['specificity'] * 100, 2)
    return  conf_matrix_dict