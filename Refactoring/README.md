# CSC466-Lab5

Students:
    Sophia Parrett (sparrett@calpoly.edu)
    Greg Bisbjerg (pbisbjer@calpoly.edu)

Programming language: 
    Python
How to run code:
    Pre-reqs:
        - numpy
        - pandas
        - nltk
    ------------------
    1) Open terminal
    2) Navigate to unzipped lab5 folder
    3) Activate  environment that contains Pre-reqs
    4) Run the respective files in the following order
    	- textVectorizer.py 
    	- RFAuthorship.py or knnAuthorship.py
    	- classfierEvaluation.py 
    ------------------

Submitted programs:
textVectorizer.py 
	Generates vectorized versions of all of the documents contained in the provided directory. 

	parser.add_argument("--text_folder", help="folder containing all test files of interest, ex. C50", type=str, required=True)
    parser.add_argument("--out_file", help="name for generated DF-IDF csv file", type=str, required=False)
    parser.add_argument("--for_RF", help="Changes Stopwords for random forest", type=bool, required=False)

    Example run
    For Random Forest
    $ python3 textVectorizer.py --text_folder C50 --for_RF True
    For KNN
    $ python3 textVectorizer.py --text_folder C50

RFAuthorship.py
	Creates a random forest and generates predictions for all of the documents for the author of interest. 
	Note: textVectorizer.py must be run prior as the file "TF_IDF.csv" is accessed to generate tailored files based on the author of interst. 

	args[1] - authorName
	args[2] - m, number of attributes per tree
	args[3] - k, number of data points used to create each tree (with replacement)
	args[4] - n, number of trees created
	args[5] - save_trees_flag (optional T or F, if none is provided false is used)
	args[6] - threshold (optional value between 0 and 1, if non is provided 0.2 is used)
	args[7] - restrictionsFile (optional file of 0 and 1 to indicate inactive columns for splitting)

	Example run
	python3 RFAuthorship.py AaronPressman 3 5 3 T 0.1

knnAuthorship.py
	


classfierEvaluation.py 
	Takes the generated files from KNN or RF and generates accuracy meterics as well as a confusion matrix

	parser.add_argument("--prediction", help="file containg predictions", type=str, required=True)

	Example run
	python3 classfierEvaluation.py --prediction rf_predictions/AaronPressman.csv

Possible Errors:
    Possible import errors if the pre-req modules are not available. The local version of this was
    tested on IOS, so errors from running in other operating systems might appear.