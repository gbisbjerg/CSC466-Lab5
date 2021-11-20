import os 
import string
import nltk
import errno

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
    lemma = nltk.wordnet.WordNetLemmatizer()
    text_file = open(file_, 'r').read()
    lines = text_file.split()
    line = [lemma.lemmatize( word.replace("'s", '').translate(str.maketrans('', '', string.punctuation)).lower()) for word in lines] 

    return line

def getGroundTruth(dirName, toCSV = True):
    list_of_files = getListOfFiles(dirName)
    groundTruthDic = {}

    if(toCSV):
        f = open("groundTruth.csv", 'w') 
        for file in list_of_files:
            author = file.split('/')[-2]
            txt_file = file.split('/')[-1]
            f.write("%s,%s\n"%(txt_file, author))

    for file in list_of_files:
        author = file.split('/')[-2]
        txt_file = file.split('/')[-1]
        groundTruthDic[txt_file] = author
    return groundTruthDic

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

