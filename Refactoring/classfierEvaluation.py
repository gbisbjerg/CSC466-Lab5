import argparse
import pandas as pd
import numpy as np
from utils import mkdir_p

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
  return f_measure

def matrix_creation(prediction_csv):
    predicitons_df = pd.io.parsers.read_csv(prediction_csv, sep="," , index_col=0)

    actual_unqiue = predicitons_df['Actual'].unique()
    predicted_unqiue = predicitons_df['Predicted'].unique()

    authors = list(np.unique( np.concatenate([actual_unqiue,predicted_unqiue]) ))
    authors.sort()

    result = [[0 for j in range(len(authors))] for i in range(len(authors))]
    for date, row in predicitons_df.T.iteritems():
        real = row['Actual']
        prediction = row['Predicted']

        real_idx = authors.index(real)
        prediction_idx = authors.index(prediction)
        result[real_idx][prediction_idx] += 1  

    ##print("result\n", result)
    final_df = pd.DataFrame(result)
    final_df.columns = authors 
    final_df.index = authors 
    return result, final_df

def output(final_lst, final_df, type_):
    result_np = np.asarray(final_lst)
    all_ = final_df.to_numpy().sum() # counts all elements (all counts in result df)
    correct = np.trace(result_np) # counts diagonol elements 
    
    all_authors = []
    all_hits = []
    all_strikes = []
    all_misses = []
    all_recall =[]
    all_precision = []
    all_fmeasure = []
    for author in final_df: 
        all_authors.append(author)
        print("\nAuthor: ",author)

        true_pos = final_df.loc[author, author]
        print("Hits: ", true_pos)
        all_hits.append(true_pos)

        false_pos = final_df[author].sum() - true_pos
        print("Strikes: ",false_pos) 
        all_strikes.append(false_pos)

        false_neg = final_df.loc[author].sum() - true_pos
        print("Misses: ", false_neg)
        all_misses.append(false_neg)
        
        recall = Recall(true_pos, false_neg)
        all_recall.append(recall)

        precision = Precision(true_pos, false_pos)
        all_precision.append(precision)

        fmeasure = Fmeasure(precision, recall)
        all_fmeasure.append(fmeasure)

    c = ['Hits', 'Strikes', 'Misses', 'Recall','Precision', 'Fmeasure' ]
    out_df = pd.DataFrame(list(zip( all_hits, all_strikes, all_misses, all_recall, all_precision, all_fmeasure)),columns =c, index=all_authors)
    print(out_df)
    mkdir_p("results")
    filename_out = 'results/results_df_' +type_
    out_df.to_csv(filename_out)


def main():
    args = parse_args()
    prediction_csv = args.prediction

    out_name = prediction_csv.split('/')[-1]

    result,final_df = matrix_creation(prediction_csv)
    output(result,final_df, out_name)


if __name__=="__main__":
    main()