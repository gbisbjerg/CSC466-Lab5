'''
Design of CSVs by row:
header: labels (strings)
0: number of attributes per label (ints)
1: Empty, except first cell (ignore)
2+: attribute for each label
NOTE: Last column of input CSVs is the "class attribute" (aka what we're trying to predict)

'''

import math
import os
import sys
import json
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
import matplotlib.pyplot as plt
import numpy as np


def unique_classes(D, class_attribute):
  return D[class_attribute].unique()


def probability(D, c, class_attribute):
  class_column = D[class_attribute]

  total_counts = class_column.value_counts(normalize=True).to_frame()
  Pr = total_counts.loc[c, class_attribute] 
  return Pr


#Entropy takes in the dataframe of all of the current points of interest and determins the entropy
#Note that this function only looks at the classification
def entropy(D, class_attribute):
  classes = unique_classes(D, class_attribute)
  total = 0
  for c in classes:
    Pr = probability(D, c, class_attribute)
    total += Pr * math.log2(Pr)
  return - total


#Finds and returns the most frequent_label in the provided DF
### NOTE not sure what happens if tied but most likely defers to alphabetical ordering
def find_most_frequent_label(D, class_attribute):
  last_column = D[class_attribute]
  total_counts = last_column.value_counts(ascending=False)
  proportion = total_counts.iloc[0] / total_counts.sum()

  return total_counts.index[0], proportion


def df_attribute_split(D, attribute):
  edges = D[attribute[0]].unique()
  edge_df = [None] * len(edges)
  for i, e in enumerate(edges):
    e_df = D.loc[D[attribute[0]] == e]
    edge_df[i] = e_df
  return edge_df


def entropy_a(D,attribute, class_attribute):
  edge_df = df_attribute_split(D, attribute)
  total_entropy = 0
  for edge in edge_df:
    edge_probability = len(edge.index) / len(D.index)
    edge_entropy = entropy(edge, class_attribute)
    total_entropy += edge_probability * edge_entropy
  return total_entropy


def probability_edge(D, attribute, e):
  attribute_column = D[attribute[0]]    # Added "[0]"

  total_counts = attribute_column.value_counts(normalize=True).to_frame()
  Pr = total_counts.loc[e, attribute[0]] # Added "[0]" 
  return Pr


def attribute_entropy(D, attribute):
  edges = D[attribute[0]].unique() # Added "[0]"
  total = 0
  for e in edges:
    Pr = probability_edge(D, attribute, e)
    total += Pr * math.log2(Pr)
  return - total


#This function selects the splitting attribute, the ratio_flag (True/False) determins if
#a information "Gain" or "Gain Ratio" is used to evalue the attributes
#Assumes A is an array of the atributes
def selectSplittingAttribute(A,D,threshold, ratio_flag, class_attribute):
  p0 = entropy(D, class_attribute)
  gains = [0] * len(A)
  gain_ratios = [0] * len(A)
  for i, a in enumerate(A):
    pa = entropy_a(D,a, class_attribute)
    gains[i] = p0 - pa
    
    if pa != 0:
      gain_ratios[i] = gains[i] / pa
    else:
      gain_ratios[i] = gains[i]

  max_gain = max(gains)
  max_gain_ratio = max(gain_ratios)

  if (max_gain > threshold and ratio_flag == False):
    return A[gains.index(max_gain)]
  elif (max_gain_ratio > threshold and ratio_flag == True):
    return A[gain_ratios.index(max_gain_ratio)]
  else:
    return None


def C45(D, A, T, threshold, class_attribute):
  num_rows = D.shape[0]
  # D will be broken up by rows, when an attribute is checked, it will be popped off A
  if num_rows < 1:
    raise Exception("C45: Input data is empty")   # Dunno if you can even call shape in this case
  # termination case 1: All the records have the same class label (expected value) therefore, create a leaf node with that value
  # all_same_label = True
  initial_label = D.iloc[0][class_attribute]  # first class label
  if len(D[class_attribute].unique()) == 1:
    T['leaf'] = {
        'decision': initial_label,
        'p': 1.0
    }
    return
  # termination case 2: No more attributes to consider, create a leaf node whose value is the pluralty class label of data in D
  if len(A) == 0:
    mfl, prop = find_most_frequent_label(D, class_attribute)
    T['leaf'] = {
        'decision': mfl,
        'p': prop
    }
    return
  # "else"
  Ao = selectSplittingAttribute(A, D, threshold, True, class_attribute)
  # print("Ao:", Ao)
  if Ao is None:
    mfl, prop = find_most_frequent_label(D, class_attribute)
    T['leaf'] = {
        'decision': mfl,
        'p': prop
    }
  else:
    node = {
        'var': Ao[0]
    }   # Will need to assign to T['node']
    edges = []   # Will need to assign to node['edges']
    unique_labels = D[Ao[0]].unique()
    for label in unique_labels:
      Dv = D[D[Ao[0]] == label] ###NOTE https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
      if Dv.shape[0] < 1:  # attribute is missing
        #mfl, prop = find_most_frequent_label(D, class_attribute)
        pass
        # currently do nothing since leaf node is made automatically
      else:
        Tv = {
            'value': label
        }
        Av = list(filter(lambda attribute: attribute != Ao, A))  # "split method"
        
        C45(Dv, Av, Tv, threshold, class_attribute)
        edges.append({
            'edge': Tv
        })
    node['edges'] = edges
    T['node'] = node
    # ghost edge will be a leaf, parser will just check edges before looking for a leaf
    mfl, prop = find_most_frequent_label(D, class_attribute)
    T['leaf'] = {
      'decision': mfl,
      'p': prop
    }


'''
C4.5 Alg takes in a new form of the dataset form all, so we must
standardize the input
'''
def format_data(dataset_loc, restrictions=None):
  '''
  Takes in raw CSV, and (optionally) an array of numbers related to the columns of the CSV.
    If a number in the array is not one, the corresponig column in the CSV is ignored
  outputs dataframe with unecessary rows removed, list of attributes, and initial tree
  '''
  df = pd.read_csv(dataset_loc)
  C = df.iloc[1][0] # class attribute

  # Drop column if number of attributes is -1 (seems to be an id we don't wanna worry about)
  num_attr_row = df.iloc[0]
  cols_to_drop = []
  initial_columns = df.columns
  # Collect which columns to drop from restriction array
  if restrictions:
    if len(restrictions) != len(initial_columns):
      raise Exception("Invalid column length in restrictions file")
    for i in range(len(restrictions)):
      if restrictions[i] < 1:
        cols_to_drop.append(initial_columns[i])
  # Collect which columns to drop from second row of training dataset
  for i, val in enumerate(num_attr_row):
    if pd.isna(val):  # missing value (the case in the mushroom dataset)
      continue
      #val = len(initial_columns[i].unique())
    elif int(val) < 0:
      cols_to_drop.append(initial_columns[i])
  df = df.drop(list(set(cols_to_drop)), axis=1)   # Drop unique column indices

  attribute_counts = df.iloc[0]   # number of different labels per attribute
  attributes = df.columns     # NOTE: Will drop the class column

  # Prepping inputs
  A = list(zip(attributes, attribute_counts))   # [(attribute_name1, num_labels1), ...]
  length_w_class_attr = len(A)
  A = [x for x in A if x[0] != C]    #Drops the class column
  length_wo_class_attr = len(A)
  if length_w_class_attr != length_wo_class_attr + 1:
    raise Exception("Class attribute was dropped due to incorrect labelling or bad restrictions")

  D = df.drop([0, 1]).reset_index(drop=True)    # drop first two rows, reset index
  T = {
      'dataset': dataset_loc
  }
  return D, A, T, C   # Dataframe of testing data, list of attribute, initial tree, class attribute


def C45_helper(trainingSetLoc, threshold, restrictionsLoc=None, printTree=True):
  '''
  Takes in command line arguments and converts them into structures that
    C45 can take in, then runs C45 on them.
  Returns the completed decision tree
  '''
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
  D, A, T, C = format_data(trainingSetLoc, restrictions) # Add class attribute
  C45(D, A, T, threshold, C) # Add class attribute
  if printTree:
    print(json.dumps(T)) # might need this to print " instead of '
    #print(T)
  return T
  # T at this stage is the completed decision tree


def C45_main():
  '''
  Usage: python InduceC45.py <TrainingSetFile.csv> <threshold> [<restrictionsFile>]
    RestrictionsFile: one line in the form "-1, 1,1,1,0,0,0,1"
  '''
  args = sys.argv
  if len(args) not in [3,4]: # if length of args is not 3 or 4
    raise Exception("Error - Usage: python InduceC45.py <TrainingSetFile.csv> <threshold> [<restrictionsFile>]")
  if len(args) == 3:
    C45_helper(args[1], args[2])
  elif len(args) == 4:
    C45_helper(args[1], args[2], args[3])
  else:
    raise Exception("I messed up my conditions - Braden")


if __name__=="__main__":
  C45_main()


# mock arguments
# dataset = "openHouses.csv"
# restrictions = "houseRestrictions.txt"
# threshold = "0.05"
# dataset_loc = data_dir + houses_dir + dataset
# restrictions_loc = data_dir + houses_dir + restrictions
# tree = C45_helper(dataset_loc, threshold, restrictions_loc, False)


# json.dumps(tree, indent=4)
# with open("file.json", 'w') as f:
#   f.write(json.dumps(tree, indent=4))


# Doesn't seem to be used
# def test_split(D, ratio):
#   '''
#   Takes in the dataframe and split ratio (0.2 -> 0.8 train / 0.2 test)
#   Returns Dtrain, Dtest
#   '''
#   Dtrain = D.sample(frac=1-ratio, random_state=1)
#   Dtest = D.drop(Dtrain.index)
#   return Dtrain, Dtest
