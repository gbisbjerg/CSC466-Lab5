
# OKAPI: SOPHIA 
# STEP 1: 

from itertools import combinations
# from cosin import bottom
from utils import getListOfFiles
from change_format import transform_df_to_forest_input
import os 
from textVectorizer import * 
import pandas as pd
import math 
import numpy as np 


from cosinDistance import normalizedTermFreq
# DELETE LATER   
pd.set_option("display.max_rows", None, "display.max_columns", None)

def document_lengths(df, short_files): 
    doc_lengths = {}
    all_doc_lengths = 0 


    for index, row in df.iterrows(): 
        
        list_ = np.array(df.loc[ index , : ])
        length = np.sum(list_)

        doc_lengths[index] = length 
        all_doc_lengths += length
        
    len_ = len(short_files)

    avdl = all_doc_lengths/ len_


    return doc_lengths, avdl, len_
        
    # sum up the row 
    # doc_lengths = {}
    # all_doc_length = 0
    # for file in listOfFiles:
    #     file_stats = os.stat(file)
    #     doc_lengths[ file.split("/")[-1]] = file_stats.st_size
    #     all_doc_length += file_stats.st_size


    # return doc_lengths,  all_doc_length/len(listOfFiles), len(listOfFiles)
        

# STEP 2 : Pair program helper function 

# okapi_helper(combos, word, word_dict, df, doc_lengths, avdl, n)
def okapi_helper(combos, word, word_dict, df, doc_lengths, avdl, n, short_files):
    # translate formula for 1 into code 
    k1 = 1 
    b = 0.75 
    k2 = 500

    #print("\nword", word, word_dict[word])
    #print("combos", combos[0], combos[1])
    dfi = len(word_dict[word].keys())
    #print("dfi", dfi)
    dlj = doc_lengths[combos[0]]
    #print("dlj", dlj)
    #print("avdl", avdl)
    #print("n", n)
    fij = df.iloc[short_files.index(combos[0])][word]  #_dict[word].get(combos[0], 0)
    ##print'fij', fij)
    fiq =  df.iloc[short_files.index(combos[1])][word]
    ##print'fiq', fiq)

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

def okapi(combos, word_dict, df, doc_lengths, avdl, n, short_files):
    file1 = combos[0]
    file2 = combos[1]
    #print("file1 ", file1)
    #print("file2 ", file2)
    # file_1 = combo[0]
    # file_2 = combo[1] 
    sum = 0

    for word in df:
        #print("word", word)
        sum +=  math.log(okapi_helper(combos, word, word_dict, df, doc_lengths, avdl, n, short_files))

    return sum 



    # sum = 0 
    # row = 0 
    # for word in matrix: #  ant ==========> balloom 
    #     sum += ln (okapi_helper(word, document_lengths, count_matrix))
    #     # formula in class 
    #     row += 1 
    # return sum 





# STEP 3 

def okapi_main(dir=None):
    # MATRIXES ALL WORDS IN ALL FILES 
    # matrix = get_matrix() # vectorize one og
    if dir ==None: 
        dir = "C50"

    # "/Users/sophiaparrett/Desktop/466/lab5/CSC466-Lab5/C50/C50test/AaronPressman"#"C50"
    #dir = "/Users/sophiaparrett/Desktop/466/lab5/CSC466-Lab5/small_test"
    #print("before short files")
    listOfFiles = getListOfFiles(dir)
    short_files = [file.split("/")[-1] for file in listOfFiles]
    print ("after short files")
    # short_files = ["1", "2", "3", "4"]



    print("before stop words")
    stop_words = put_stop_words_in_list(filename="./general/word_file_count.csv")
    print("after stop words")
    print("before word_dict\n")
    word_dict = create_word_count_dict(dir, stop_words)
    print("after word_dict\n")

    print("before main df")
    df = create_df(word_dict)
    df.to_csv("knn/counts.csv", sep=',')

    transform_df_to_forest_input("knn/counts.csv", dir, "AlexanderSmith", "knn/formatted_counts.csv")
    print("after main df")
    #print("df", df)

    print("before doc lengths")
    doc_lengths, avdl, n = document_lengths(df, short_files)
    #print("n", n)
    #print("doc_lengths", doc_lengths)
    #print("advl", avdl)
    print("after doc lengths")

    k1 = 1 
    b = 0.75 
    k2 = 500
    P2 = {} # part 2 of the equation 
    P3 = {}  # part 3 of the equation 

    all_words = np.array(df.columns.values.tolist())

    step1 = [len(word_dict[word].keys()) for word in all_words] # np.log(np.array([len(word_dict[word].keys()) for word in all_words]))
    #print("step1 first", step1)
    step1 = np.array(step1)
    step1 = np.log(((n - step1 + 0.5) / (step1 + 0.5)))
    #print("n", n)

    #print("\nstep1", step1)

    print("before calculations")

    for filename in df.rows:
        #print("\nfilename", filename)
        fij = np.array(df.loc[ filename , : ])
        #print("fij", fij)
        dlj = doc_lengths[filename]
        #print("dlj", dlj)
        step2 =  ((k1 + 1)*fij) / ((k1 * (1 - b + (b * (dlj/avdl))))+ fij) 
        #print("step2", step2)
        P2[filename] = step2 

        fiq = np.array(df.loc[ filename , : ])
        #print("fiq", fiq)
        step3 =  ((k2+1) * fiq) / (k2 + fiq)
        #print("step3", step3)
        P3[filename] = step3 

    #print("step1", step1)
    #print("P2", P2)
    #print("P3", P3)
    


    num_files = len(short_files)

    print("after calculations")
    
    df_distances = [[0 for j in range(num_files)] for i in range(num_files)]
    i = 0 
    j = 0


    for i in range(num_files): 
        #print("files:{} ".format(short_files[i]))
        for j in range(num_files): 
            #print'step1', step1)
            step2 = P2[short_files[j]]
            #print("step2", step2)
            step3 = P3[short_files[i]]
            #print("step3", step3)
            first_half = np.multiply(step1, step2)
            all_multiplied = np.multiply(first_half, step3)
            #print("all multiplt", all_multiplied)

            final_distance = np.sum(all_multiplied)
            df_distances[i][j] = final_distance
        #    break 



    final_df = pd.DataFrame(df_distances)
    
    with open("knn/distance_filenames.csv", 'w') as f: 
        write = csv.writer(f) 
        write.writerow(short_files) 

    with open("knn/okapi_distances.csv", 'w') as f: 
        write = csv.writer(f) 
        write.writerow(short_files) 
        write.writerows(df_distances) 









