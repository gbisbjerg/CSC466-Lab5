
# OKAPI: SOPHIA 
# STEP 1: 

from itertools import combinations
from cosin import bottom
from utils import getListOfFiles
import os 
from textVectorizer import * 
import pandas as pd
import math 

# DELETE LATER   
pd.set_option("display.max_rows", None, "display.max_columns", None)

def document_lengths(listOfFiles): 
    doc_lengths = {}
    all_doc_length = 0
    for file in listOfFiles:
        file_stats = os.stat(file)
        doc_lengths[ file.split("/")[-1]] = file_stats.st_size
        all_doc_length += file_stats.st_size


    return doc_lengths,  all_doc_length/len(listOfFiles), len(listOfFiles)
        

# STEP 2 : Pair program helper function 

# okapi_helper(combos, word, word_dict, df, doc_lengths, avdl, n)
def okapi_helper(combos, word, word_dict, df, doc_lengths, avdl, n):
    # translate formula for 1 into code 
    k1 = 1 
    b = 0.75 
    k2 = 500

    #print("\nword", word, word_dict[word])
    dfi = len(word_dict[word].keys())
    #print("dfi", dfi)
    dlj = doc_lengths[combos[0]]
    #print("dlj", dlj)
    #print("avdl", avdl)
    #print("n", n)
    fij = word_dict[word].get(combos[0], 0)
    #print('fij', fij)
    fiq = word_dict[word].get(combos[1], 0)
    #print('fiq', fiq)

    first = (n-dfi+0.5)/ (dfi+0.5)
    second = ((k1 + 1)*fij) / ((k1 * (1 - b + (b * (dlj/avdl)) ))+ fij)
    third = ((k2+1) * fiq) / (k2 + fiq)

    sol = first * second * third
    #print("sol", sol)
    if sol == 0: 
        return 1
    return sol





    # dfi   number of documents that contain word  look_at_dict len(word_dict[word].values.keys()) in vecotization.py 
    # dlj   length of doc => look up in document lengths 
    # fij   frequency of word in document 1   look up in count matrix 
    # fiq   frequency of word in document 2   look up in count matrix 


# okapi(combo, word_dict, df, doc_lengths, avdl, n)

def okapi(combos, word_dict, df, doc_lengths, avdl, n):
    file1 = combos[0]
    file2 = combos[1]
    #print("file1 ", file1)
    #print("file2 ", file2)
    # file_1 = combo[0]
    # file_2 = combo[1] 
    sum = 0

    for word in df:
        sum +=  math.log(okapi_helper(combos, word, word_dict, df, doc_lengths, avdl, n))

    return sum 



    # sum = 0 
    # row = 0 
    # for word in matrix: #  ant ==========> balloom 
    #     sum += ln (okapi_helper(word, document_lengths, count_matrix))
    #     # formula in class 
    #     row += 1 
    # return sum 





# STEP 3 

def main():
    # MATRIXES ALL WORDS IN ALL FILES 
    # matrix = get_matrix() # vectorize one og
    dir = "C50"# "/Users/sophiaparrett/Desktop/466/lab5/CSC466-Lab5/C50/C50test/AaronPressman"#"C50"
    #dir = "/Users/sophiaparrett/Desktop/466/lab5/CSC466-Lab5/small_test"
    listOfFiles = getListOfFiles(dir)
    short_files = [file.split("/")[-1] for file in listOfFiles]
    # short_files = ["1", "2", "3", "4"]
    doc_lengths, avdl, n = document_lengths(listOfFiles)
    

    
    stop_words = put_stop_words_in_list(filename="word_file_count.csv")
    word_dict = create_word_count_dict(dir, stop_words)
    #print("word_dict\n", word_dict)
    df = create_df(word_dict)

    file_combinations = [comb for comb in combinations(short_files, 2)]
    #print("file combos", file_combinations)
    
    distance_df = pd.DataFrame(data=None, index=short_files, columns=short_files, dtype=None, copy=False)
    #print("df\n", distance_df)
    ##print("combos", file_combinations)


    for combo in file_combinations: 
        print("combo", combo)
        #df.insert(combo[0], combo[1], okapi(combo))
        distance_df.loc[combo[0]][combo[1]] = okapi(combo, word_dict, df, doc_lengths, avdl, n)

    
    #print("df\n", distance_df)
    distance_df.to_csv("okapi_small_test_output", sep=',')

    #     Result: 
    #       1    2    3    4
    # 1  NaN    1    1    1
    # 2  NaN  NaN    1    1
    # 3  NaN  NaN  NaN    1
    # 4  NaN  NaN  NaN  NaN


    



   


    # combinations = []  # all combos of all documents 
    
    # for combo in combinations: 
    #     okapi(combo)



       # d1 d2 d3 d4 d5 
    # d1 
    # d2    distances between each document 
    # d3 
    # d4




    





main()
