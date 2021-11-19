import math
import os
import sys
import json
import pandas as pd
from utils import mkdir_p
from cosinDistance import rf_format_TF_IDF
from textVectorizer import vectorizer_main
pd.options.mode.chained_assignment = None  # default='warn' https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
import matplotlib.pyplot as plt
import numpy as np
from rf_InduceC45 import format_data, C45
from rf_classifier import classifier
import copy
import random
import json

import time

def AttributeRandSelect(A, NumAttributes):
    # random.seed(rand_seed)
    return random.sample(A, NumAttributes)
# print(AttributeRandSelect([[1,2],[1,3],[1,4]], 2, 109))

def FolderName(save_trees, save_trees_file_name):
    existing_trees = 0
    tree_files = []

    author_name = save_trees_file_name[0]
    attributes_used = str(save_trees_file_name[1])
    data_points_used = str(save_trees_file_name[2])
    threshold_used = str(save_trees_file_name[3])
    folder_name = author_name + "_" + attributes_used + "_" + data_points_used  + "_" + threshold_used
    forest_folder = 'forest/' + folder_name

    if(save_trees):
        file = mkdir_p(forest_folder)
        tree_files = os.listdir(forest_folder)
        existing_trees = len(tree_files)
    return existing_trees, tree_files, forest_folder

def ForestCreation(author_df, non_author_df, initial_T, A, C, NumAttributes, NumDataPoints, NumTrees, threshold, save_trees, save_trees_file_name):
    existing_trees, tree_files, forest_folder = FolderName(save_trees, save_trees_file_name)
    #Dataset Selection/Behavior
    forest = [None] * NumTrees

    trees = 0
    #If there are enough trees then skip, or only make the desired remainder
    for i, tree_file in enumerate(tree_files):
        # print(tree_file)
        if(trees == NumTrees - 1):
            break
        tree_loc = forest_folder + "/" + tree_file

        with open(tree_loc, 'r') as myfile:
            data=myfile.read()
            forest[i] = json.loads(data)
        trees += 1

    #Load the stored trees 
    for i in range(trees, NumTrees):
        non_author_df = non_author_df.sample(frac=1).head(400)
        D = pd.concat([author_df, non_author_df])

        T = copy.deepcopy(initial_T)
        selected_attributes = AttributeRandSelect(A, NumAttributes)
        selected_data = D.sample(NumDataPoints,replace=True) #Samples datapoints WITH replacement
        C45(selected_data, selected_attributes, T, threshold, C) 
        forest[i] = T
        #print("tree ", i)
        if(save_trees):
            file = open(forest_folder + '/tree{0}.txt'.format(existing_trees),'w+')
            json. dump(T, file, indent = 4)
            existing_trees += 1
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


def rf_format_data(df_file, threshold, restrictionsLoc):
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
    return D, A, initial_T, C

def CreateResultsDF(df_shuffled, C):
    res_DF = pd.DataFrame(columns = ['Predicted', 'Actual'])
    actual_column = df_shuffled[C]
    res_DF = pd.DataFrame(index=df_shuffled.index)
    res_DF['Actual'] = actual_column.copy()
    return res_DF


def RandomForest(authorName, NumAttributes, NumDataPoints, NumTrees, save_trees_flag = "F", threshold = 0.2, restrictionsLoc = None):
    if(save_trees_flag == "T"):
        save_trees = True
    else:
        save_trees = False

    print("Start of Run")
    start = time.time()

    df_file_csv = rf_format_TF_IDF(authorName, "C50")
    D, A, initial_T, C = rf_format_data(df_file_csv, threshold, restrictionsLoc)

    # Create the data pool, selecting the 100 from the authoer and 400 from other authors
    df_shuffled = D.sample(frac=1) #Shuffles the dataframe in case it is ordered by a value
    author_df = df_shuffled.loc[df_shuffled[C] == authorName]
    non_author_df = df_shuffled.loc[df_shuffled[C] != authorName]

    forest_start = time.time()
    print("Forest Start", forest_start - start)
    # ForestCreation
    save_trees_file_name = [authorName, NumAttributes, NumDataPoints, threshold]
    forest = ForestCreation(author_df, non_author_df, initial_T, A, C, NumAttributes, NumDataPoints, NumTrees, threshold, save_trees, save_trees_file_name)
    forest_end = time.time()
    print("Forest End", forest_end - start)

    # Forest Testing with all documents
    classification_set = pd.concat([author_df, non_author_df])
    predicted_values, correct, incorrect = ForestClassifier(classification_set, forest, C)
    classifier_end = time.time()
    print("Classifier End", classifier_end - start)
    res_DF = CreateResultsDF(classification_set, C)
    res_DF['Predicted'] = pd.Series(predicted_values, index=classification_set.index)
    confusion_matrix = pd.crosstab(res_DF['Actual'], res_DF['Predicted'], rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix, "\n")


'''
Expects the following args, 
args[1] - authorName
args[2] - m, number of attributes per tree
args[3] - k, number of data points used to create each tree (with replacement)
args[4] - n, number of trees created
args[5] - save_trees_flag (optional T or F, if none is provided false is used)
args[6] - threshold (optional value between 0 and 1, if non is provided 0.2 is used)
args[7] - restrictionsFile (optional file of 0 and 1 to indicate inactive columns for splitting)

Usage: python randomForest.py <Author Name> <DataSetFile.csv> <Attributes per tree> <data points per tree> <number of trees>
    [<rand_seed>] [<threshold>] [<restrictionFile>]
'''
def random_forest_main():

    args = sys.argv
    if len(args) not in [5,6,7,8]:
        raise Exception("Error - Usage: python rf_randomForest.py <Attributes per tree> <data points per tree> <number of trees>\
    [<save trees>] [<threshold>] [<restrictionFile>]")
    if len(args) == 5:
        RandomForest(args[1], int(args[2]), int(args[3]), int(args[4])) 
    elif len(args) == 6:
        RandomForest(args[1], int(args[2]), int(args[3]), int(args[4]), args[5]) 
    elif len(args) == 7:
        RandomForest(args[1], int(args[2]), int(args[3]), int(args[4]), args[5], float(args[6])) 
    elif len(args) == 8:
        RandomForest(args[1], int(args[2]), int(args[3]), int(args[4]), args[5], float(args[6]), args[7]) 
    else:
        raise Exception("I messed up my conditions - Greg :)")

if __name__=="__main__":
    random_forest_main()

