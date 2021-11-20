import sys
from numpy.core.arrayprint import _make_options_dict 
import pandas as pd 
import math 
import json 
import numpy as np
import random
from pandas.core.tools.numeric import to_numeric
from utils import *
import csv 
from collections import Counter

authors = ["AaronPressman", "AlanCrosby", "AlexanderSmith", "BenjaminKangLim", "BernardHickey", "BradDorfman", "DarrenSchuettler", "DavidLawder", "EdnaFernandes", "EricAuchard", "FumikoFujisaki", "GrahamEarnshaw", "HeatherScoffield", "JanLopatka", "JaneMacartney", "JimGilchrist", "JoWinterbottom", "JoeOrtiz", "JohnMastrini", "JonathanBirt", "KarlPenhaul", "KeithWeir", "KevinDrawbaugh", "KevinMorrison", "KirstinRidley", "KouroshKarimkhany", "LydiaZajc", "LynneO'Donnell", "LynnleyBrowning", "MarcelMichelson", "MarkBendeich", "MartinWolk", "MatthewBunce", "MichaelConnor", "MureDickie", "NickLouth", "PatriciaCommins", "PeterHumphrey", "PierreTran", "RobinSidel", "RogerFillion", "SamuelPerry", "SarahDavison", "ScottHillis", "SimonCowell", "TanEeLyn", "TheresePoletti", "TimFarrand", "ToddNissen", "WilliamKazer"]

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def classifier_knn(k, type_):
    ground_truth = getGroundTruth("C50", toCSV = False)
    filenames = []
    distance_df = None 
    if type == "okapi": 
        with open( "knn/distance_filenames.csv", newline='') as f:
            reader = csv.reader(f)
            filenames = next(reader)

        distance_df = pd.read_csv("knn/distance_filenames.csv")

    else: # cosin 
        with open( "knn/full_cosin_C50", newline='') as f:
            reader = csv.reader(f)
            filenames = next(reader)
            filenames.remove('')

        distance_df = pd.read_csv("knn/full_cosin_C50")
        distance_df = distance_df.iloc[: , 1:]



    np.fill_diagonal(distance_df.values, -100000)
    # num_files = len(filenames)
    result = [[0 for j in range(50)] for i in range(50)]
    #authors = [ground_truth[x] for x in filenames]

    ##print("K", k)
    for i in range(len(distance_df)):
        #print("i", i)
        distance_list = list(distance_df.loc[ i, : ])
        #print("1", distance_list)
        distance_list_np = np.array(distance_list)
        #print("2", distance_list_np)
        ##print("K", k, type(k))
        largest = distance_list_np.argsort()[::-1][:k]
        #print("3", largest)
        predicted_list = np.array([ground_truth[filenames[x]] for x in largest])
        #print("predicted list", predicted_list)
        prediction = Most_Common(predicted_list)
        #print("prediction", prediction)
        prediction_idx = authors.index(prediction)
        #print("prediction idx", prediction_idx)
        real = ground_truth[filenames[i]]
        #print("real", real)
        real_idx = authors.index(real)
        ##print("real idx", real_idx)
        result[real_idx][prediction_idx] += 1  
        # exit()

    ##print("result\n", result)
    final_df = pd.DataFrame(result)
    final_df.columns = authors 
    final_df.index = authors 
    


    

    ##print(final_df)
    
    return result, final_df


def Recall(truePos, falseNeg):
  recall = (round((truePos / (truePos + falseNeg)), 2))
  print("Recall: " + str(recall))
  return recall

def Precision(truePos, falsePos):
  precision = round((truePos / (truePos + falsePos)), 2)
  print("Precision: " + str(precision))
  return precision

def Fmeasure(precision, recall):
  f_measure = round (( 2 * ((precision * recall) / (precision + recall))), 2)
  print("f-measure: " + str(f_measure))



def output(final_lst, final_df):
    result_np = np.asarray(final_lst)
    all_ = final_df.to_numpy().sum() # counts all elements (all counts in result df)
    correct = np.trace(result_np) # counts diagonol elements 

    for author in final_df: 
        print("\nAuthor: ",author)
        true_pos = final_df.loc[author, author]
        print("Hits: ", true_pos)
        false_pos = final_df[author].sum() - true_pos
        print("Strikes: ",false_pos) 
        false_neg = final_df.loc[author].sum() - true_pos
        print("Misses: ", false_neg)
        
        recall = Recall(true_pos, false_neg)
        precision = Precision(true_pos, false_pos)
        Fmeasure(precision, recall)


        

    
    print("\nTotal Number Documents Correctly Predicted: ", correct) 
    print("Total Number Documents Incorrectly Predicted:: ", all_ - correct)
    print("Overall Accuracy: " ,(round(correct/all_, 2))) 


#     For each author in the dataset, output the total number of hits (correctly predicted documents), strikes
# (false positives predicted) and misses (document written by the author, which were not attributed to the
# author).
# • For each author in the dataset report the precision, the recall, and the f1-measure of the KNN procedure.





 
def main():
    #type_ = input("Insert Type:(okapi or cosin)")
    type_ =  "okapi" #"cosin"

    for k in range(2,3):
        print("Type:", type_)
        print("\nK =", k)
        
    
        lst_of_lsts, df = classifier_knn( k, type_)
        output(lst_of_lsts, df)
        
        df.to_csv('knn/out.csv')


        

main()

