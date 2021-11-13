import argparse
import os
import csv
from utils import * 

'''
This program parese all of the provided text files in the folder of interest and generates a word
frequency list based on the number of files that contain the word. This list is sorted by ascending order and 
and returned in the form of a CSV file.  

Example run:
>> python StopWords.py --text_folder C50  
>> python StopWords.py --text_folder C50  --out_file c50_frequency_words.csv
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_folder", help="folder containing all test files of interest, ex. C50", type=str, required=True)
    parser.add_argument("--out_file", help="name for generated frequency csv file", type=str, required=False)
    args = parser.parse_args()
    return args

'''
    For the given path, get the List of all files in the directory tree 
    Source - https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
'''
   

def getOutFile(args):
    out_file = ""
    if (args.out_file):
        out_file = args.out_file
    return out_file



def findUniqueWords(clean_text):
    unique = []
    for word in clean_text:
        if word not in unique:
            unique.append(word)
    return unique

def containsNumber(word):
    for character in word:
        if character.isdigit():
            return True
    return False

def updateWordFileCount(word_file_count, unique_words):
    for word in unique_words:
        if word_file_count and word in word_file_count:
            word_file_count[word] += 1
        else:
            word_file_count[word] = 1
    return word_file_count

def makeWordCount_CSV(out_file, word_file_count):
    if (out_file == ""):
        out_file = "word_file_count.csv"

    with open(out_file, 'w') as f:
        for key, value in sorted(word_file_count.items(),key=lambda item: item[1]):
            f.write("%s,%s\n"%(key, value))

def main():
    #Get the arguments provided from the user
    args = parse_args()
    text_folder = args.text_folder
    out_file = getOutFile(args)

    #Generates a list of all the paths to the text files contained 
    listOfFiles = getListOfFiles(text_folder)
    
    word_file_count = {}
    for file in listOfFiles:
        clean_text = textCleaning(file)
        unique_words = findUniqueWords(clean_text)
        word_file_count = updateWordFileCount(word_file_count, unique_words)

    word_file_count.pop("", None)
    makeWordCount_CSV(out_file, word_file_count)

    #0.015 * len()




if __name__ == "__main__":
    main()