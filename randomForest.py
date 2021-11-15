import math
import os
import sys
import json
import pandas as pd

from change_format import change_format_main
from textVectorizer import vectorizer_main
pd.options.mode.chained_assignment = None  # default='warn' https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
import matplotlib.pyplot as plt
import numpy as np
from InduceC45 import format_data, C45
from classifier import classifier
import copy
import random

def CreateResultsDF(df_shuffled, C):
    res_DF = pd.DataFrame(columns = ['Predicted', 'Actual'])
    actual_column = df_shuffled[C]
    res_DF = pd.DataFrame(index=df_shuffled.index)
    res_DF['Actual'] = actual_column.copy()
    return res_DF

def AttributeRandSelect(A, NumAttributes, rand_seed):
    random.seed(rand_seed)
    return random.sample(A, NumAttributes)
# print(AttributeRandSelect([[1,2],[1,3],[1,4]], 2, 109))

def ForestCreation(D, initial_T, A, C, NumAttributes, NumDataPoints, NumTrees, rand_seed, threshold ):
    #Dataset Selection/Behavior
    forest = [None] * NumTrees
    for i in range(0,NumTrees):
        
        T = copy.deepcopy(initial_T)

        selected_attributes = AttributeRandSelect(A, NumAttributes, rand_seed)
        selected_data = D.sample(NumDataPoints,replace=True) #Samples datapoints WITH replacement
        C45(selected_data, selected_attributes, T, threshold, C) 
        forest[i] = T
        print("tree: ", i)
    return forest

def ForestClassifier(test_set, forest, C): 
    #Create dataframe to store predictions
    D_predictions = pd.DataFrame(columns = ['Actual'])
    actual_column = test_set[C]
    D_predictions = pd.DataFrame(index=test_set.index)
    D_predictions['Actual'] = actual_column.copy()

    for i, tree in enumerate(forest):
        group_predicted_values, correct, incorrect = classifier(test_set, tree, C, True)
        D_predictions[i] = group_predicted_values

    ##Ignore the correct answer column and but the most common answer for the row
    guesses = D_predictions.drop('Actual',axis=1)
    D_predictions['Predictions'] = guesses.mode(axis=1).iloc[:, 0]

    D_predictions['Check'] = np.where((D_predictions['Actual'] == D_predictions['Predictions']), True, False)
    try:
        correct = D_predictions.Check.value_counts().loc[True]
    except KeyError:
        correct = 0
    try:
        incorrect = D_predictions.Check.value_counts().loc[False]
    except KeyError:
        incorrect = 0

    #Take column and make into a row
    forest_predicted_values = np.asarray(D_predictions['Predictions']).tolist()
    return forest_predicted_values, correct, incorrect 


def RandomForest(df_file, NumAttributes, NumDataPoints, NumTrees, rand_seed = 1, threshold = 0.2, restrictionsLoc = None):
    # vectorizer_main()
    change_format_main()
    restrictions = None
    try:
        threshold = float(threshold)
        if restrictionsLoc:  # parsing restrictions file here to just pass an array
            with open(restrictionsLoc, 'r') as rf:
                lines = rf.readlines() 
                if len(lines) != 1:
                    raise Exception("restrictionsFile formatted incorrectly. Format ex: \"-1,1,1,0,0,1\"")
                restrictions = [int(x.strip()) for x in lines[0].split(',')]  # turns file text input into an array of integers
    except Exception as e:
        print(e)
        return
    D, A, initial_T, C = format_data(df_file, restrictions)

    ###Evaluation
    #Shuffles the dataframe in case it is ordered by a value
    number_folds = 1#10
    df_shuffled = D.sample(frac=1, random_state= rand_seed)
    training_set = df_shuffled

    #Creation of results DataFrame
    res_DF = CreateResultsDF(df_shuffled, C)
    predicted_values = []
    rows = len(df_shuffled)

    max_group_size = math.ceil(rows/number_folds)   
    
    #Values used to walk through the cross-section area
    min = 0
    max = max_group_size

    group_correct = []
    group_incorrect = []
    group_accuracy = []
    group_error = []
    #Traversal of the various sections
    for group in range(0, number_folds):
        test_set = df_shuffled[min:max]
        df_top = df_shuffled[0:min]
        df_bottom = df_shuffled[max:rows]
        training_set = pd.concat([df_top, df_bottom])

        #Call to generate forest
        forest = ForestCreation(D, initial_T, A, C, NumAttributes, NumDataPoints, NumTrees, rand_seed, threshold )
        group_predicted_values, correct, incorrect = ForestClassifier(test_set, forest, C)
        predicted_values += group_predicted_values

        total_count = correct + incorrect
        group_correct.append(correct)
        group_incorrect.append(incorrect)
        try: 
            group_accuracy.append( correct/total_count )
            
        except ZeroDivisionError as e:
            ### FIX AVERAGE ACCURACY 
            group_accuracy.append( 0.0)

        try: 
            group_error.append( incorrect/total_count )
        
        except ZeroDivisionError as e:
            ### FIX AVERAGE ACCURACY 
            group_error.append( 0.0)

        min = max
        max += max_group_size

    res_DF['Predicted'] = pd.Series(predicted_values, index=df_shuffled.index)
    confusion_matrix = pd.crosstab(res_DF['Actual'], res_DF['Predicted'], rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix, "\n")

    #Metrics
    res_DF['Check'] = np.where((res_DF['Predicted'] == res_DF['Actual']), True, False)
    correct_predictions = res_DF.Check[res_DF.Check==True].count()

    total_count = sum(group_correct) + sum(group_incorrect)
    print("Overall accuracy: " + str(sum(group_correct)/total_count))
    print("Overall error rate: " + str(sum(group_incorrect)/total_count), "\n")

    print("Average accuracy: " + str( sum(group_accuracy) / len(group_accuracy) ))
    print("Average error rate: " + str( sum(group_error) / len(group_error) ))

#Expects the following args, 
#args[1] - Dataset File
#args[2] - m, number of attributes per tree
#args[3] - k, number of data points used to create each tree (with replacement)
#args[4] - n, number of trees created
#args[5] - rand_seed (optional integer, if none is provided 1 is used)
#args[6] - threshold (optional value between 0 and 1, if non is provided 0.2 is used)
#args[7] - restrictionsFile (optional file of 0 and 1 to indicate inactive columns for splitting)
def random_forest_main():
    '''
    Usage: python randomForest.py <DataSetFile.csv> <Attributes per tree> <data points per tree> <number of trees>
    [<rand_seed>] [<threshold>] [<restrictionFile>]
    '''
    args = sys.argv
    if len(args) not in [3, 4, 5, 6]: # if length of args is not 3, 4, or 5
        raise Exception("Error - Usage: python randomForest.py <DataSetFile.csv> <Attributes per tree> <data points per tree> <number of trees>\
    [<rand_seed>] [<threshold>] [<restrictionFile>]")
    if len(args) == 5:
        RandomForest(args[1], int(args[2]), int(args[3]), int(args[4])) 
    elif len(args) == 6:
        RandomForest(args[1], int(args[2]), int(args[3]), int(args[4]), int(args[5])) 
    elif len(args) == 7:
        RandomForest(args[1], int(args[2]), int(args[3]), int(args[4]), int(args[5]), float(args[6])) 
    elif len(args) == 8:
        RandomForest(args[1], int(args[2]), int(args[3]), int(args[4]), int(args[5]), float(args[6]), args[7]) 
    else:
        raise Exception("I messed up my conditions - Greg :)")

if __name__=="__main__":
    random_forest_main()

