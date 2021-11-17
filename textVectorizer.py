



import os
import re
import string
import pandas as pd
from utils import getListOfFiles, textCleaning
import csv

# Important: https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-then-filling-it
all_filenames = []

def create_word_count_dict(dir_, stop_words): 
    # #print("textVectorizer.py: create_word_count_dict")
    # fix OS walk 
    # directory = os.fsencode(dir_)
    unique_words = {}
    
    # goes through all of the txt files in a directory 
    # for file in os.listdir(directory):
    #     filename = os.fsdecode(file)
    listOfFiles = getListOfFiles(dir_)
    
    i = 0
    for file in listOfFiles:
        
        lines = textCleaning(file)
        filename = file.split("/")[-1]
        #print("filename", i)
        # #print("filename", filename)
        all_filenames.append(filename)
        ##print("filename", filename)
        for w in lines: 
            #for w in line: 
            # gets rid of punctuation in a word, makes the word lowercase 
                # NOTE: Still need to get rid of 's at end of word will most likely need to modify this ^ 
            if (len(w) > 0) and (w not in stop_words):
                if w in list(unique_words.keys()):
                    if filename in list(unique_words[w].keys()): 
                        if unique_words[w][filename] < 500: 
                            unique_words[w][filename] += 1 
                    else: 
                        unique_words[w][filename] = 1 

                else: 
                    unique_words[w] = {filename:1}

            else:
                continue
        i+=1 

    return unique_words


def initialize_2d_array(data, num_rows, num_columns): 
	return [[data for i in range(num_columns)] for j in range(num_rows)] 

def create_df(word_count_dict):
    all_words = list(word_count_dict.keys())
    num_files = len(all_filenames)
    num_words = len(all_words)
    matrix = initialize_2d_array(0, num_files, num_words)

    for key in word_count_dict:
        all_file_counts = word_count_dict[key]
        for file_ in list(all_file_counts.keys()): 
            index_of_word = all_words.index(key)
            index_of_file = all_filenames.index(file_)
            matrix[index_of_file][index_of_word] = word_count_dict[key][file_]

    df = pd.DataFrame(matrix, index=all_filenames)
    df.columns = all_words
    df.rows = all_filenames
    return df 

def put_stop_words_in_list(filename): 
    file = open(filename, "r")
    csv_reader = csv.reader(file)
    lists_from_csv = []
    ones = []
    
    for row in csv_reader:
        if row[1] == "1": 
            ones.append(row[0])
        lists_from_csv.append(row)

    num_stop_words = -1 * int(0.0155 * len(lists_from_csv))
    stop_words =  [word[0] for word in  lists_from_csv[num_stop_words:]]
    # #print("stop words", stop_words + ones)
    return stop_words + ones




def convert_df_counts_to_ratios(): 
    pass 
    # NOTE: still need to do this with numpy 

def vectorizer_main(): 
    directory = "C50"
    stop_words = put_stop_words_in_list(filename="./general/word_file_count.csv")
    word_dict = create_word_count_dict(directory, stop_words)
    df = create_df(word_dict)
    df.to_csv("./general/c50_word_counts_in", sep=',')
