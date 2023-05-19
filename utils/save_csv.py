import pandas as pd
import numpy as np

def save_result(defect_count, good_count, total_count, weekdays):
    """
    This function save the detected results of spur gear as CSV file

    Input:
    preds - list of predicted labels
    weekdays - weekdays of the production running
    Return - csv file 
    """

    qc_results = []
    qc_results += [defect_count, good_count, total_count]
    qc_results.insert(0, np.nan)

    try:
        df = pd.read_csv('./runs/csv/inspection_result.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Weekdays', 'Defects Teeth', 'Good Teeth', 'Total Products'])
    # print(len(df))
    # print(qc_results)

    while len(df) <= 5:
        df = df.append(pd.Series(qc_results, index=df.columns), ignore_index=True)
        #print(len(df))
        if len(df)==1:
            df.loc[0, 'Weekdays'] = weekdays[0]
        elif len(df)==2:
            df.loc[1, 'Weekdays'] = weekdays[1]
        elif len(df)==3:
            df.loc[2, 'Weekdays'] = weekdays[2]
        elif len(df)==4:
            df.loc[3, 'Weekdays'] = weekdays[3]
        elif len(df)==5:
            df.loc[4, 'Weekdays'] = weekdays[4]
        elif len(df)==6:
            df.loc[5, 'Weekdays'] = weekdays[5]
        break

    return df.to_csv('./runs/csv/inspection_result.csv', index=False)