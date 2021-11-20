import pandas as pd 
import numpy as np
from utils import *
import argparse 
from collections import Counter
from cosinDistance import cosin_main

authors = ["AaronPressman", "AlanCrosby", "AlexanderSmith", "BenjaminKangLim", "BernardHickey", "BradDorfman", "DarrenSchuettler", "DavidLawder", "EdnaFernandes", "EricAuchard", "FumikoFujisaki", "GrahamEarnshaw", "HeatherScoffield", "JanLopatka", "JaneMacartney", "JimGilchrist", "JoWinterbottom", "JoeOrtiz", "JohnMastrini", "JonathanBirt", "KarlPenhaul", "KeithWeir", "KevinDrawbaugh", "KevinMorrison", "KirstinRidley", "KouroshKarimkhany", "LydiaZajc", "LynneO'Donnell", "LynnleyBrowning", "MarcelMichelson", "MarkBendeich", "MartinWolk", "MatthewBunce", "MichaelConnor", "MureDickie", "NickLouth", "PatriciaCommins", "PeterHumphrey", "PierreTran", "RobinSidel", "RogerFillion", "SamuelPerry", "SarahDavison", "ScottHillis", "SimonCowell", "TanEeLyn", "TheresePoletti", "TimFarrand", "ToddNissen", "WilliamKazer"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_vectorizer_file", help="file containing text vectorization", type=str, required=True)
    parser.add_argument("--k", help="k value", type=int, required=True)
    parser.add_argument("--metric", help="Choose: 1 for Okapi, 2 for Cosine", type=int, required=True)
    args = parser.parse_args()
    return args

def document_lengths(df): 
    doc_lengths = {}
    all_doc_lengths = 0 
    count = 0 

    for index, row in df.iterrows(): 
        list_ = np.array(df.loc[ index , : ])
        length = np.sum(list_)
        doc_lengths[index] = length 
        all_doc_lengths += length
        count += 1 
        
    avdl = all_doc_lengths/ count 
    return doc_lengths, avdl, count


def short_files(df): 
    #print("short files", )
    all_files = []
    for index, row in df.iterrows(): 
        #print("index", index)
        all_files.append(index)
    # print("ALL FILS", all_files)
    # f = [file_.split("/")[-1] for file_ in all_files]
    # print(f)
    #print(all_files, type(all_files))
    return all_files


def okapi_main(df, short_files):
    doc_lengths, avdl, n = document_lengths(df)
    

    k1 = 1 
    b = 0.75 
    k2 = 500
    P2 = {} 
    P3 = {}  

    step1 = list(np.count_nonzero(df, axis=0))
    step1 = np.array(step1)
    step1 = np.log(((n - step1 + 0.5) / (step1 + 0.5)))

    for filename in df.iterrows():
        filename = filename[0]
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


    num_files = len(short_files)
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
    return final_df

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

# DocumentId,  Actual Predicted
# document123, John.   John
# document124, Sam.     John
# document345, Kate.    Kate
def knn(distance_df, k, filenames):
    #print(type(filenames))
    #print("distance df", distance_df)
    ground_truth = getGroundTruth("C50", toCSV = False)
    #print("ground truth", ground_truth)
    actual_list = []
    predict_list = []
    np.fill_diagonal(distance_df.values, -100000)
    

    for filename in distance_df.iterrows():
        filename = filename[0]
        #print("filename\n", filename)
        distance_list = list(distance_df.loc[ filename, : ])
        distance_list_np = np.array(distance_list)
        largest = distance_list_np.argsort()[::-1][:k]

        predicted_list = np.array([ground_truth[filenames[x]] for x in largest])
        prediction = Most_Common(predicted_list)
        predict_list.append(prediction)
        if type(filename) == int: 
            real = ground_truth[filenames[filename]]
        else: 
            real = ground_truth[filename]
        actual_list.append(real)

    #print("predict", predict_list, len(predict_list))
    #print("actual" , actual_list, len(actual_list))
    #print("filenames", filenames, len(filenames))
    predictions_df  = pd.DataFrame(
    {'Filename': filenames,
     'Actual': actual_list,
     'Predicted': predict_list
    })
    return predictions_df



def main(): 
    args = parse_args()
    vector_filename = args.text_vectorizer_file
    df = pd.read_csv(vector_filename, index_col=0)
    abrv_files = short_files(df)
    metric = ''

    if args.metric == 1: 
        df = okapi_main(df, abrv_files)
        metric = "okapi"
        

    else: 
        cosin_main(vector_filename)
        df = pd.read_csv("full_cosin_C50", index_col=0)
        metric = "cosin"

    predictions_df = knn(df, args.k, abrv_files)
    predictions_df = predictions_df.set_index('Filename')
    predictions_df.to_csv("predictions_{}_out.csv".format(metric))
    #print(predictions_df)





    


main()


    
    # with open("knn/distance_filenames.csv", 'w') as f: 
    #     write = csv.writer(f) 
    #     write.writerow(short_files) 

    # with open("knn/okapi_distances.csv", 'w') as f: 
    #     write = csv.writer(f) 
    #     write.writerow(short_files) 
    #     write.writerows(df_distances) 



#     For each author in the dataset, output the total number of hits (correctly predicted documents), strikes
# (false positives predicted) and misses (document written by the author, which were not attributed to the
# author).
# • For each author in the dataset report the precision, the recall, and the f1-measure of the KNN procedure.





 
# def main():
#     args = parse_args()
#     type_int =  int(input("Pick: \n\t1 = Okapi\n\t2 for Cosin Distance") )
#     if type_int == 1: 
#         type_ = "okapi"
#     else: 
#         type_ = "cosin"


#     dir = "C50"

#     if type_ == "okapi": 
#         okapi_main(dir)
#     else: 
#         cosin_main()
        

#     for k in range(2,3):
#         print("Type:", type_)
#         print("\nK =", k)
        
    
#         lst_of_lsts, df = classifier_knn( k, type_)
#         output(lst_of_lsts, df, type_)
        
        # df.to_csv('knn/out.csv')


        



