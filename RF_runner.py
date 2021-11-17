

import csv 
from rf_randomForest import RandomForest

with open("TESTING_RF.csv", newline='') as f_in: #read from
    reader = csv.reader(f_in)

    row = next(reader)
    while (row):
        print("Attribites {} Datapoints {} Trees {}, Threshold {}".format(row[0], row[1], row[2], row[3]))
        RandomForest("AaronPressman", int(row[0]), int(row[1]), int(row[2]), "T", float(row[3]))
        row = next(reader)


    
    
