


import csv 
import utils 
import os 


# example out 
# Color,Size,Act,Age,Inflated
# 2,2,2,2,2
# Inflated
# YELLOW,SMALL,STRETCH,ADULT,T
# YELLOW,SMALL,STRETCH,CHILD,T


# words 
#  0 1



# words yellow balloom cow   author 
#   0                          name 
# 
# 
#  


def transform_df_to_forest_input(csv_in, dirName, selected_author):
    #print(os.listdir("./"))
    truth = utils.getGroundTruth(dirName)
    print("FUNCTION: transform_df_to_forest_input")
    

    f_out = open("./general/change_format_forest_in", 'w') # writting to
    with open(csv_in, newline='') as f_in: #read from
        reader = csv.reader(f_in)
        write = csv.writer(f_out) 

        row1 = next(reader)
        #print("row1", row1)
        row1[0] = "doc_name"
        row1.append("AUTHOR")

        write.writerow(row1) 

        listOfZero = [0] * len(row1)
        listOfZero[0] = -1
        write.writerow(listOfZero) 
        
        write.writerow(["AUTHOR"]) 
        i = 0 
        for row in reader:
            print("row: " , i)
            author_name = truth[row[0]]
            if author_name != selected_author: 
                author_name = "Not Author"
            row.append(author_name)
            write.writerow(row)
            i+=1 


    
def change_format_main(): 
    # othher function 
    transform_df_to_forest_input(csv_in="./general/c50_word_counts_in", dirName="C50", selected_author="AaronPressman")





        
