import os 
import string

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        if(entry != ".DS_Store"):
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory 
            if os.path.isdir(fullPath):
                allFiles = allFiles + getListOfFiles(fullPath)
            else:
                allFiles.append(fullPath)

    return allFiles  


def textCleaning(file_):
    
    text_file = open(file_, 'r').read()
    lines = text_file.split()
    line = [word.replace("'s", '').translate(str.maketrans('', '', string.punctuation)).lower() for word in lines] 

    return line

