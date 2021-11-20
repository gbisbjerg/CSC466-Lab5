import csv 
import utils 
import os 


def transform_df_to_forest_input(csv_in, dirName, selected_author, csv_out):
    #Creates dictionary to link the documents and authors 
    truth = utils.getGroundTruth(dirName)
    
    f_out = open(csv_out, 'w') # writting to
    with open(csv_in, newline='') as f_in: #read from
        reader = csv.reader(f_in)
        write = csv.writer(f_out) 

        row1 = next(reader)
        row1[0] = "doc_name"
        row1.append("AUTHOR")

        write.writerow(row1) 

        listOfZero = [0] * len(row1)
        listOfZero[0] = -1
        write.writerow(listOfZero) 
        
        write.writerow(["AUTHOR"]) 
        i = 0 
        for row in reader:
            author_name = truth[row[0]]
            if author_name != selected_author: 
                author_name = "Not Author"
            row.append(author_name)
            write.writerow(row)
            i+=1 
