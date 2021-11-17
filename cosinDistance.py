from textVectorizer import put_stop_words_in_list, create_word_count_dict, create_df
from change_format import transform_df_to_forest_input
from utils import mkdir_p
import pandas as pd
from StopWords import GenerateWordCount
import argparse
import numpy as np
import csv
import sys
import math
from utils import getListOfFiles, textCleaning
from itertools import combinations
import os.path

np.set_printoptions(threshold=sys.maxsize)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_folder", help="folder containing all test files of interest, ex. C50", type=str, required=True)
    args = parser.parse_args()
    return args

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

def cosin_distance(TF_IDF_DF, similarity_csv_name):
    f_out = open(similarity_csv_name, 'w')
    indexNamesArr = list(TF_IDF_DF.index.values)
    header = ','.join(['%s' % num for num in indexNamesArr])
    f_out.write(header + '\n')

    total = [None] * len(TF_IDF_DF.index)

    j = 0
    for file in indexNamesArr:
        print("Calculating cosin for ", file)
        TF_IDF_quere = np.array(TF_IDF_DF.loc[file, :]).astype(float)

        cosin_bot_q = np.sum(TF_IDF_quere**2)
        
        i = j
        for index, row in TF_IDF_DF.iloc[j:].iterrows():
            TF_IDF_document =  np.asarray(row, dtype=np.float64) #np.array(row).astype(float)

            # cosin_top = np.sum(TF_IDF_quere * TF_IDF_document)
            cosin_top = TF_IDF_quere.ravel().dot(TF_IDF_document.ravel())

            cosin_bottom = math.sqrt( np.sum(TF_IDF_document**2) * cosin_bot_q )
            
            total[i] = np.true_divide(cosin_top, cosin_bottom)
            i += 1

        row_norm = file + ',' + (','.join(['{}'.format(num) for num in total])) + '\n'
        f_out.write(row_norm)

        total[j] = 'N/A'
        j += 1

### vvv For random forest - This should be relocated to a better spot
def stop_words_7000(wordFileCount): 
    file = open(wordFileCount, "r")
    csv_reader = csv.reader(file)
    words_from_csv = []
    
    for row in csv_reader:
        words_from_csv.append(row)

    offEnd = len(words_from_csv) - 70
    stop_words_start = [word[0] for word in words_from_csv[offEnd:]]
    stop_words_end = [word[0] for word in words_from_csv[0:offStart]]
    stop_words = stop_words_start + stop_words_end
    print(len(stop_words))
    return stop_words


def rf_format_TF_IDF(author, directory = "C50"):
    file = mkdir_p("forest/rf")
    #Get list of files from the directory
    print("Step 1: Getting list of files in directory")
    listOfFiles = getListOfFiles(directory)
    short_files = [file.split("/")[-1] for file in listOfFiles]
    NUMBER_OF_FILES_RUN = len(listOfFiles)

    #Generate initial vectorized DF
    print("Step 2: Creating initial vectorized DF")
    if( not os.path.isfile("forest/rf/word_file_count.csv") ):
        GenerateWordCount(directory, "forest/rf/word_file_count.csv")
    if( not os.path.isfile("forest/rf/rf-vectorDF.csv") ):
        stop_words = stop_words_7000("forest/rf/word_file_count.csv")   ###CHANGE THIS
        word_dict = create_word_count_dict(directory, stop_words)
        vectorized_df = create_df(word_dict)
        vectorized_df.to_csv("forest/rf/rf-vectorDF.csv", sep=',')

    #Generate the normalized DF
    print("Step 3: Creating the normalized DF")
    if( not os.path.isfile("forest/rf/rf--normalizedTermDF.csv") ):
        normalizedTermDF = normalizedTermFreq("forest/rf/rf-vectorDF.csv", "forest/rf/rf--normalizedTermDF.csv")    

    #Generate the TF-IDF DF which contains the wij values
    print("Step 4: Creating the TF-IDF DF")
    if( not os.path.isfile("forest/rf/rf-TF_IDF.csv") ):
        idf_dic = inverseDocumentFrequency_dic(NUMBER_OF_FILES_RUN, "forest/rf/word_file_count.csv")
        Create_TF_IDF_DF("forest/rf/rf--normalizedTermDF.csv", idf_dic, "forest/rf/rf-TF_IDF.csv")

    #Transforming CSV to forest input
    print("Step 5: Converting to forest input format")
    if( not os.path.isfile("forest/rf/rf-TF_IDF{}.csv".format(author)) ):
        transform_df_to_forest_input("forest/rf/rf-TF_IDF.csv", directory, author, "forest/rf/rf-TF_IDF{}.csv".format(author))

    return "forest/rf/rf--normalizedTermDF{}.csv".format(author)

### ^^^ For random forest - This should be relocated to a better spot

def main():
    print("Step 1: Getting args from user")
    #Get the arguments provided from the user
    args = parse_args()
    text_folder = args.text_folder

    #Get list of files from the directory
    print("Step 2: Getting list of files in directory")
    listOfFiles = getListOfFiles(text_folder)
    short_files = [file.split("/")[-1] for file in listOfFiles]
    NUMBER_OF_FILES_RUN = len(listOfFiles)

    #Generate initial vectorized DF
    print("Step 3: Creating initial vectorized DF")
    stop_words = put_stop_words_in_list(filename="word_file_count.csv")
    word_dict = create_word_count_dict(text_folder, stop_words)
    vectorized_df = create_df(word_dict)
    vectorized_df.to_csv("cosin-vectorDF", sep=',')
    
    #Generate the normalized DF
    print("Step 4: Creating the normalized DF")
    normalizedTermDF = normalizedTermFreq('cosin-vectorDF', "cosin-normalizedTermDF")

    #Determin the inverse document frequence DF
    print("Step 5: Creating the inverse document frequence DF")
    idf_dic = inverseDocumentFrequency_dic(NUMBER_OF_FILES_RUN)
    
    # idf_file = open("cosin-idf", "w")
    # writer = csv.writer(idf_file)
    # for key, value in idf_dic.items():
    #     writer.writerow([key, value])
    # idf_file.close()

    #Generate the TF-IDF DF which contains the wij values
    print("Step 6: Creating the TF-IDF DF")
    Create_TF_IDF_DF("cosin-normalizedTermDF", idf_dic, 'cosin-TF_IDF')

    #Generate the similiarty DF
    print("Step 7: Creating the Similarity DF")

    TF_IDF_DF = pd.read_csv('cosin-TF_IDF', index_col=0)
    cosin_distance(TF_IDF_DF,"cosin_C50")

    # similarity_df.to_csv("cosin_C50", sep=',')
    # print("DONE! :)")
