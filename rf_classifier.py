'''
Classifier (complete just not tested):
--------------
Might need to change to input CSV, but for now, will assume that we just get the Dtest dataframe
(Kinda confused because he wants us to output the C45 tree to stdout, which is weird because this function needs it as input either as a dict or a json file)

inputs: Dtest, jsonTree
outputs: confusion matrix
'''
import math
import os
import sys
import json
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
import matplotlib.pyplot as plt
import numpy as np
from rf_InduceC45 import format_data

def classifier(D, tree, class_attribute, ret_pred=False):
  '''
  Input:
    D - dataframe containing test data with named columns
    tree - decision tree dictionary
    class_attribute - string
    ret_pred - Condition whether or not to only return the predicted class labels
      Will also not print results to screen if ret_pred is true
  '''
  predicted_class_labels = []
  for i, row in D.iterrows():
    predicted_class_labels.append(parser(row, tree))
  actual = D[class_attribute]
  total_classified = len(predicted_class_labels)
  num_correct, num_incorrect = checkAccuracy(actual.tolist(), predicted_class_labels)
  if ret_pred:
    return predicted_class_labels, num_correct, num_incorrect
  print(f"Total number of records classified: {total_classified}")
  print(f"Total number of records correctly classified: {num_correct}")
  print(f"Total number of records incorrectly classified: {num_incorrect}")
  print(f"Overall accuracy / Error Rate: {100*num_correct/total_classified}% / {100*num_incorrect/total_classified}%")
  return num_correct, num_incorrect
    

def parser(row, tree):
  '''
  parses decision tree based on row data

  input: row from test data (row), an edge of the tree (tree)
  output: predicted value for row
  '''
  
  if 'node' in tree: # at node, continue down the tree
    node = tree['node']
    deciding_attribute = node['var']
    deciding_value = row.loc[deciding_attribute]
    edges = node['edges']
    for edge in edges:
      if deciding_value == edge['edge']['value']: # Follow this edge
        return parser(row, edge['edge'])
    # if you get here no path was found, return the ghost edge's value
    return tree['leaf']['decision']
  else: # at leaf, return its decision
    return tree['leaf']['decision']



def checkAccuracy(actual, expected, matrix=False):
  '''
  actual: list, expected: List
  '''
  if len(actual) != len(expected):
    raise Exception("Error checkAccuracy: different sized lists")
  num_correct, num_incorrect = 0, 0
  for i in range(len(actual)):
    actual_label = actual[i]
    expected_label = expected[i]
    if actual_label == expected_label:
      num_correct += 1
    else:
      num_incorrect += 1
    
    # for decision matrix matrix[actual_label][expected_label] += 1
    # can make a decision matrix with a 2d list or a 2d dictionary....
  return num_correct, num_incorrect


def classifier_helper(csv_loc, json_loc):
  '''
  Input: locations of the testing dataset and json decision tree
    Assumes testing dataset is in the same form as the CSV input to InduceC45
  '''
  D, A, T, C = format_data(csv_loc) # ignoring A and T
  # print(json_loc)
  with open(json_loc, 'r', encoding="utf-16") as json_file:
    # print(json_file.read())
    tree = json.load(json_file)
  classifier(D, tree, C)
  


def classifier_main():
  '''
  Usage: python classify.py <CSV file (test data)> <JSON file (tree)>
  '''
  args = sys.argv
  if len(args) != 3:
    raise Exception("Error - python classify.py <CSV file (test data)> <JSON file (tree)>")
  classifier_helper(args[1], args[2])


if __name__=="__main__":
  classifier_main()
