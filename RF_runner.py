

import csv 
from rf_randomForest import random_forest_main

with open("TESTING_RF.csv", newline='') as f_in: #read from
    reader = csv.reader(f_in)
    row = next(reader)
    print("Attribites {} Datapoints {} Trees {}, Threshold {}".format(row[0], row[1], row[2], row[3]))
    random_forest_main(row[0], row[1], row[2], "T", row[3])


    
    
