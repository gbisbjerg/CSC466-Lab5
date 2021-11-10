



import os
import re
import string
import pandas as pd

# Important: https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-then-filling-it
all_filenames = []

def create_word_count_dict(dir_, stop_words): 
    directory = os.fsencode(dir_)
    unique_words = {}
    
    # goes through all of the txt files in a directory 
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"): 
            full_filename = dir_ + "/" + filename
            str_ = open(full_filename).read()
            lines = str_.split()
            all_filenames.append(filename)
           
            
            for word in lines:
                # gets rid of punctuation in a word, makes the word lowercase 
                w = word.translate(str.maketrans('', '', string.punctuation)).lower()
                # NOTE: Still need to get rid of 's at end of word will most likely need to modify this ^ 
                if (len(w) > 0) and (w not in stop_words):
                    if w in list(unique_words.keys()):
                        if filename in list(unique_words[w].keys()): 
                            unique_words[w][filename] += 1 
                        else: 
                            unique_words[w][filename] = 1 

                    else: 
                        unique_words[w] = {filename:1}

        else:
            continue

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
    str_ = open(filename).read()
    lines = str_.split()
    return lines


def convert_df_counts_to_ratios(): 
    pass 
    # NOTE: still need to do this with numpy 

def main(): 
    directory = "/Users/sophiaparrett/Desktop/466/lab5/CSC466-Lab5/C50/C50test/AaronPressman"
    stop_words = put_stop_words_in_list(filename="stop_words.txt")
    word_dict = create_word_count_dict(directory, stop_words)
    df = create_df(word_dict)
    print(df)
    


main()