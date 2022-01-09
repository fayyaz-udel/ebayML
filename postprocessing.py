import csv

import pandas as pd
import numpy as np


def calculate_delivery_date(tsv_path):
    quiz_label = pd.read_csv("./output/quiz_result.csv", header=None).round()[0].to_list()
    print("is NAN number: " + str(np.isnan(quiz_label).sum()))
    quiz = pd.read_csv(tsv_path, sep="\t")
    quiz_data = pd.to_datetime(quiz["acceptance_scan_timestamp"].str.slice(0, 10))
    average = round(quiz["carrier_min_estimate"] + quiz["carrier_max_estimate"] / 2)
    out_file = open('./output/output.tsv', 'w+', newline='')
    tsv_writer = csv.writer(out_file, delimiter='\t')

    for index, value in quiz_data.items():
        if pd.isna(quiz_label[index]):
            quiz_label[index] = average[index]
        tsv_writer.writerow([str(15000001 + index), str(value + pd.Timedelta(days=quiz_label[index]))[:10]])
        if index % 100000 == 0:
            print(index)
    out_file.flush()
    out_file.close()


