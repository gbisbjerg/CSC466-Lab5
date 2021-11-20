import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", help="file containg predictions", type=str, required=True)
    args = parser.parse_args()
    return args

def Recall(truePos, falseNeg):
  recall = truePos / (truePos + falseNeg)
  print("Recall: " + str(recall))
  return recall

def Precision(truePos, falsePos):
  precision = truePos / (truePos + falsePos)
  print("Precision: " + str(precision))
  return precision

def Fmeasure(precision, recall):
  f_measure = 2 * ((precision * recall) / (precision + recall))
  print("f-measure: " + str(f_measure))

def main():
    args = parse_args()
    prediction_csv = args.prediction

    results = pd.io.parsers.read_csv(prediction_csv, sep="," , index_col=0)

    confusion_matrix = pd.crosstab(results['Actual'], results['Predicted'], rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix, "\n")

    authorName = "AaronPressman"
    precision = Precision(confusion_matrix.at["AaronPressman", "AaronPressman"], confusion_matrix[authorName]["Not Author"])
    recall = Recall(confusion_matrix[authorName][authorName], confusion_matrix["Not Author"][authorName])
    Fmeasure(precision, recall)

if __name__=="__main__":
    main()