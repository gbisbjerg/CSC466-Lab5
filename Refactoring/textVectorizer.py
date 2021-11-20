import argparse
import os
import re
import string
import pandas as pd
from utils import getListOfFiles, textCleaning, getGroundTruth
import csv
import numpy as np
import math

all_filenames = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_folder", help="folder containing all test files of interest, ex. C50", type=str, required=True)
    parser.add_argument("--out_file", help="name for generated DF-IDF csv file", type=str, required=False)
    parser.add_argument("--for_RF", help="Changes Stopwords for random forest", type=bool, required=False)
    args = parser.parse_args()
    return args

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

def initialize_2d_array(data, num_rows, num_columns): 
    return [[data for i in range(num_columns)] for j in range(num_rows)] 

def findUniqueWords(clean_text):
    unique = []
    for word in clean_text:
        if word not in unique:
            unique.append(word)
    return unique

def create_word_count_dict(dir_, stop_words): 
    unique_words = {}
    
    listOfFiles = getListOfFiles(dir_)
    
    i = 0
    for file in listOfFiles:
        lines = textCleaning(file)
        filename = file.split("/")[-1]
        all_filenames.append(filename)
        for w in lines: 
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

def makeWordCount_CSV(out_file, word_file_count):
    if (out_file == ""):
        out_file = "word_file_count.csv"

    with open(out_file, 'w') as f:
        for key, value in sorted(word_file_count.items(),key=lambda item: item[1]):
            f.write("%s,%s\n"%(key, value))

def updateWordFileCount(word_file_count, unique_words):
    for word in unique_words:
        if word_file_count and word in word_file_count:
            word_file_count[word] += 1
        else:
            word_file_count[word] = 1
    return word_file_count

def GenerateWordCount(directory, out_file):
    #Generates a list of all the paths to the text files contained 
    listOfFiles = getListOfFiles(directory)
    
    word_file_count = {}
    for file in listOfFiles:
        clean_text = textCleaning(file)
        unique_words = findUniqueWords(clean_text)
        word_file_count = updateWordFileCount(word_file_count, unique_words)

    word_file_count.pop("", None)
    makeWordCount_CSV(out_file, word_file_count)

def stop_words_7000(wordFileCount): 
    file = open(wordFileCount, "r")
    csv_reader = csv.reader(file)
    words_from_csv = []
    
    for row in csv_reader:
        words_from_csv.append(row)

    offStart = 39887
    offEnd = len(words_from_csv) - 70
    stop_words_start = [word[0] for word in words_from_csv[offEnd:]]
    stop_words_end = [word[0] for word in words_from_csv[0:offStart]]
    stop_words = stop_words_start + stop_words_end
    print(len(stop_words))
    return stop_words

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
    return stop_words + ones

def normalizedTermFreq(vectorized_csv_name, normalized_csv_name):
    f_out = open(normalized_csv_name, 'w')

    with open(vectorized_csv_name, newline='') as f_in:
        reader = csv.reader(f_in)

        # Write over the header
        row1 = next(reader) 
        header = ','.join(['%s' % num for num in row1])
        f_out.write(header + '\n')

        for row in reader:
            file_name = row[0]
            data_row = np.array(row[1:]).astype(float)
            row_max = data_row.max()
            normalized_data_row = np.true_divide(data_row, row_max)

            row_norm = file_name + ',' + (','.join(['%f' % num for num in normalized_data_row])) + '\n'
            f_out.write(row_norm)
    f_out.close()

def inverseDocumentFrequency_dic(document_count = 5000, file_count_csv = "word_file_count.csv"):
    word_file_count_dic = {}

    with open(file_count_csv) as fc:
        reader = csv.reader(fc, delimiter=",")
        for row in reader:
            word_file_count_dic[str(row[0])] = math.log2(document_count / int(row[1]))
    return word_file_count_dic

def Create_TF_IDF_DF(normalized_csv_name, word_file_count_dic, TF_IDF_csv_name):
    f_out = open(TF_IDF_csv_name, 'w')

    with open(normalized_csv_name, newline='') as f_in:
        reader = csv.reader(f_in)
        # Write over the header
        row1 = next(reader) 
        header = ','.join(['%s' % num for num in row1])
        f_out.write(header + '\n')

        #Create an array to be used for idf values
        row1 = row1[1:]
        idf = [0] * len(row1)
        for i, word in enumerate(row1):
            idf[i] = word_file_count_dic[word]
        idf = np.array(idf)

        #Pull each row and generate and save TF IDF values
        for row in reader:
            file_name = row[0]
            data_row = np.array(row[1:]).astype(float)

            TF_IDF_DF = np.multiply(data_row, idf)

            row_norm = file_name + ',' + (','.join(['%f' % num for num in TF_IDF_DF])) + '\n'
            f_out.write(row_norm)

def TF_IDF_main():
    args = parse_args()
    directory = args.text_folder
    out_file = "TF_IDF.csv"
    if(args.out_file):
        out_file = args.out_file
    for_RF = False
    if(args.for_RF):
        for_RF = args.for_RF

    print("Getting list of files in directory")
    listOfFiles = getListOfFiles(directory)
    short_files = [file.split("/")[-1] for file in listOfFiles]
    NUMBER_OF_FILES_RUN = len(listOfFiles)

    #Generate initial vectorized DF
    print("Step 2: Creating initial vector of word counts")
    if( not os.path.isfile("word_file_count.csv") ):
        GenerateWordCount(directory, "word_file_count.csv")
    if( not os.path.isfile("vectorDF.csv") ):
        if(for_RF):
            stop_words = stop_words_7000("word_file_count.csv")
        else:
            stop_words = put_stop_words_in_list("word_file_count.csv")
        word_dict = create_word_count_dict(directory, stop_words)
        vectorized_df = create_df(word_dict)
        vectorized_df.to_csv("vectorDF.csv", sep=',')

    #Generate the normalized DF
    print("Step 3: Creating the normalized DF")
    if( not os.path.isfile("normalizedTermDF.csv") ):
        normalizedTermDF = normalizedTermFreq("vectorDF.csv", "normalizedTermDF.csv")    

    #Generate the TF-IDF DF which contains the wij values
    print("Step 4: Creating the TF-IDF DF")
    if( not os.path.isfile("TF_IDF.csv") ):
        idf_dic = inverseDocumentFrequency_dic(NUMBER_OF_FILES_RUN, "word_file_count.csv")
        Create_TF_IDF_DF("normalizedTermDF.csv", idf_dic, "TF_IDF.csv")

    #Generate the ground truth file
    print("Step 5: Creating the ground truth file")
    getGroundTruth(directory)

if __name__=="__main__":
    TF_IDF_main()
