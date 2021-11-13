from textVectorizer import put_stop_words_in_list, create_word_count_dict, create_df
import pandas as pd
import argparse
import numpy as np
import csv
import sys
import math

np.set_printoptions(threshold=sys.maxsize)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_folder", help="folder containing all test files of interest, ex. C50", type=str, required=True)
    args = parser.parse_args()
    return args

def normalizedTermFreq(vectorized_df):
    normalizedTermDF = vectorized_df.copy(deep=True)
    for index, row in normalizedTermDF.iterrows():
        data_row = row.to_numpy()
        row_max = data_row.max()
        normalizedTermDF.loc[index, :] = np.true_divide(data_row, row_max)

    return normalizedTermDF

def inverseDocumentFrequency_dic(document_count = 5000, file_count_csv = "word_file_count.csv"):
    word_file_count_dic = {}

    with open(file_count_csv) as fc:
        reader = csv.reader(fc, delimiter=",")
        for row in reader:
            word_file_count_dic[str(row[0])] = math.log2(document_count / int(row[1]))
    return word_file_count_dic

def Create_TF_IDF_DF(normalizedTermDF, word_file_count_dic):
    TF_IDF_DF = normalizedTermDF.copy(deep=True)

    for column_name in TF_IDF_DF.columns.values.tolist():
        column_name = str(column_name)
        idf = word_file_count_dic[column_name]
        TF_IDF_DF[column_name] = TF_IDF_DF[column_name] * idf

    return TF_IDF_DF

def cosin_distance(TF_IDF_DF, d1_name, d2_name):
    #TOP
    TF_IDF_d1 = TF_IDF_DF.loc[d1_name, :].to_numpy()
    TF_IDF_d2 = TF_IDF_DF.loc[d2_name, :].to_numpy()

    cosin_top = np.sum(TF_IDF_d1 * TF_IDF_d2)
    
    #BOTTOM
    cosin_bottom = math.sqrt( np.sum(np.square(TF_IDF_d1)) ) * math.sqrt( np.sum(np.square(TF_IDF_d2)) )

    return cosin_top / cosin_bottom

def main():
    NUMBER_OF_FILES_RUN = 5000
    #Get the arguments provided from the user
    args = parse_args()
    text_folder = args.text_folder

    stop_words = put_stop_words_in_list(filename="word_file_count.csv")
    word_dict = create_word_count_dict(text_folder, stop_words)
    vectorized_df = create_df(word_dict)
    vectorized_df.to_csv("vectorDF", sep='\t')

    normalizedTermDF = normalizedTermFreq(vectorized_df)
    normalizedTermDF.to_csv("normalizedTermDF", sep='\t')

    idf_dic = inverseDocumentFrequency_dic(NUMBER_OF_FILES_RUN)
    idf_file = open("idf.csv", "w")

    writer = csv.writer(idf_file)
    for key, value in idf_dic.items():
        writer.writerow([key, value])
    idf_file.close()


    TF_IDF_DF = Create_TF_IDF_DF(normalizedTermDF, idf_dic)
    TF_IDF_DF.to_csv("TF_IDF_DF", sep='\t')

    print( cosin_distance(TF_IDF_DF, "517526newsML.txt", "42764newsML.txt") )
    print( cosin_distance(TF_IDF_DF, "517526newsML.txt", "517526newsML.txt") )

main()