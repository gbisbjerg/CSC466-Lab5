

<<<<<<< HEAD
import csv 
from rf_randomForest import RandomForest

with open("TESTING_RF.csv", newline='') as f_in: #read from
    reader = csv.reader(f_in)

    row = next(reader)
    while (row):
        print("Attribites {} Datapoints {} Trees {}, Threshold {}".format(row[0], row[1], row[2], row[3]))
        RandomForest("AaronPressman", int(row[0]), int(row[1]), int(row[2]), "T", float(row[3]))
        row = next(reader)
=======
from csv import reader 
from rf_randomForest import RandomForest
import os 

first_half = ['AaronPressman', 'AlanCrosby', 'AlexanderSmith', 'BenjaminKangLim', 'BernardHickey', 'BradDorfman', 'DarrenSchuettler', 'DavidLawder', 'EdnaFernandes', 'EricAuchard', 'FumikoFujisaki', 'GrahamEarnshaw', 'HeatherScoffield', 'JanLopatka', 'JaneMacartney', 'JimGilchrist', 'JoWinterbottom', 'JoeOrtiz', 'JohnMastrini', 'JonathanBirt', 'KarlPenhaul', 'KeithWeir', 'KevinDrawbaugh', 'KevinMorrison', 'KirstinRidley']
second_half = ['KouroshKarimkhany', 'LydiaZajc', "LynneO'Donnell", 'LynnleyBrowning', 'MarcelMichelson', 'MarkBendeich', 'MartinWolk', 'MatthewBunce', 'MichaelConnor', 'MureDickie', 'NickLouth', 'PatriciaCommins', 'PeterHumphrey', 'PierreTran', 'RobinSidel', 'RogerFillion', 'SamuelPerry', 'SarahDavison', 'ScottHillis', 'SimonCowell', 'TanEeLyn', 'TheresePoletti', 'TimFarrand', 'ToddNissen', 'WilliamKazer']

for author in second_half:  
    print("\nauthor ", author)
    RandomForest(authorName=author,NumAttributes= 500 ,NumDataPoints= 300, NumTrees= 50, save_trees_flag="T", threshold= 0.1)


>>>>>>> ac3f55b55e5febfcefb6617800ce623b3b31135b


    
