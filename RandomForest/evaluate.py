import math
import os
import sys
import json
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
import matplotlib.pyplot as plt
import numpy as np
from InduceC45 import format_data, C45
from classifier import classifier
import copy

'''
Overall confusiong matrix: matrix for each call
recall, prevision, pf, and f-measure
Overall and average accuracy and error rate of prediction
'''

def ConfusionMatrix(truePos, trueNeg, falsePos, falseNeg):
  print("Confusion Matrix")
  print("True Positive: " + str(truePos))
  print("True Negative: " + str(trueNeg))
  print("False Positive: " + str(falsePos))
  print("False Negative: " + str(falseNeg) + "\n")

def Recall(truePos, falseNeg):
  recall = truePos / (truePos + falseNeg)
  print("Recall: " + str(recall))
  return recall

def Precision(truePos, falsePos):
  precision = truePos / (truePos + falsePos)
  print("Precision: " + str(precision))
  return precision

def Pf(trueNeg, falsePos):
  pf = falsePos / (falsePos + trueNeg)
  print("pf: " + str(pf))

def Fmeasure(precision, recall):
  f_measure = (2 * precision * recall) / precision + recall
  print("f-measure: " + str(f_measure))

def CreateResultsDF(df_shuffled, C):
  res_DF = pd.DataFrame(columns = ['Predicted', 'Actual'])
  actual_column = df_shuffled[C]
  res_DF = pd.DataFrame(index=df_shuffled.index)
  res_DF['Actual'] = actual_column.copy()
  return res_DF

def OverallAccuracy(truePos, trueNeg, falsePos, falseNeg):
  accuracy = (truePos + trueNeg) / (falsePos + falseNeg + truePos + trueNeg)
  print("Overall accuracy: " + str(accuracy))

def OverallErrorRate(truePos, trueNeg, falsePos, falseNeg):
  error_rate = (falsePos + falseNeg ) / (falsePos + falseNeg + truePos + trueNeg)
  print("Overall error rate: " + str(error_rate))

#This function takes df (dataframe), n (number of slices), rand_seed (a seed for a random number generator), 
#type (a string to determin the tree generation), and returns the accuracy
def CrossValidation(df_file, n, rand_seed = 1, threshold = 0.2, restrictionsLoc = None):
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

  df, A, initial_T, C = format_data(df_file, restrictions)
  #Shuffles the dataframe in case it is ordered by a value
  df_shuffled = df.sample(frac=1, random_state= rand_seed)
  training_set = df_shuffled

  #Creation of results DataFrame
  res_DF = CreateResultsDF(df_shuffled, C)
  predicted_values = []
  rows = len(df_shuffled)
  
  group_correct = []
  group_incorrect = []
  group_accuracy = []
  group_error = []
  if (n == 0 or n == 1):
    T = copy.deepcopy(initial_T)
    C45(training_set, A, T, threshold, C) 
    group_predicted_values, correct, incorrect = classifier(training_set, T, C, True)
    group_correct.append(correct)
    group_incorrect.append(incorrect)
    predicted_values += group_predicted_values
  else:
    if (n == -1):
      max_group_size = 1 #All but one cross validation
      i = rows
    else:
      max_group_size = math.ceil(rows/n)   
      i = n
  
    #Values used to change cross-section area
    min = 0
    max = max_group_size

    #Traversal of the various sections
    for group in range(0, i):
      test_set = df_shuffled[min:max]
      df_top = df_shuffled[0:min]
      df_bottom = df_shuffled[max:rows]
      training_set = pd.concat([df_top, df_bottom])
      #C45 Call to generate JSON tree
      T = copy.deepcopy(initial_T)
      C45(training_set, A, T, threshold, C) 
      group_predicted_values, correct, incorrect = classifier(test_set, T, C, True)
      predicted_values += group_predicted_values

      total_count = correct + incorrect
      group_correct.append(correct)
      group_incorrect.append(incorrect)
      group_accuracy.append( correct/total_count )
      group_error.append( incorrect/total_count )

      min = max
      max += max_group_size

  res_DF['Predicted'] = pd.Series(predicted_values, index=df_shuffled.index)
  # res_DF = res_DF.value_counts().to_frame('counts').reset_index()

  confusion_matrix = pd.crosstab(res_DF['Actual'], res_DF['Predicted'], rownames=['Actual'], colnames=['Predicted'])
  print(confusion_matrix, "\n")

  total_count = sum(group_correct) + sum(group_incorrect)
  print("Overall accuracy: " + str(sum(group_correct)/total_count))
  print("Overall error rate: " + str(sum(group_incorrect)/total_count), "\n")

  if( not(n == 0 or n == 1) ):
    print("Average accuracy: " + str( sum(group_accuracy) / len(group_accuracy) ))
    print("Average error rate: " + str( sum(group_error) / len(group_error) ))

#Expects the following args, 
#args[1] - Training File
#args[2] - n (number of folds)
#args[3] - rand_seed (optional integer, if non is provided 1 is used)
#args[4] - threshold (optional value between 0 and 1, if non is provided 0.2 is used)
#args[4] - restrictionsFile (optional file of 0 and 1 to indicate inactive columns for splitting)
def evaluation_main():
  '''
  Usage: python evaluate.py <TrainingSetFile.csv> <number of folds> [<rand_seed>] [<threshold>]
  '''
  args = sys.argv
  if len(args) not in [3, 4, 5, 6]: # if length of args is not 3, 4, or 5
    raise Exception("Error - Usage: python evaluate.py <TrainingSetFile.csv> <number of folds> [<rand_seed>] [<threshold>]")
  if len(args) == 3:
    CrossValidation(args[1], int(args[2]))
  elif len(args) == 4:
    CrossValidation(args[1], int(args[2]), int(args[3]))
  elif len(args) == 5:
    CrossValidation(args[1], int(args[2]), int(args[3]), float(args[4])) 
  elif len(args) == 6:
    CrossValidation(args[1], int(args[2]), int(args[3]), float(args[4], args[5])) 
  else:
    raise Exception("I messed up my conditions - Braden")

if __name__=="__main__":
  evaluation_main()