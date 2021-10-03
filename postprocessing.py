import pandas as pd
import csv


def calculate_delivery_date():
    quiz_label = pd.read_csv("./data/quiz_result.csv", header=None).round()[0].to_list()
    quiz_data = pd.read_csv("./data/quiz.tsv", sep="\t")
    quiz_data = pd.to_datetime(quiz_data["acceptance_scan_timestamp"].str.slice(0, 10))
    out_file = open('./output.tsv', 'w+', newline='')
    tsv_writer = csv.writer(out_file, delimiter='\t')

    for index, value in quiz_data.items():
        tsv_writer.writerow([str(15000001 + index), str(value + pd.Timedelta(days=quiz_label[index]))[:10]])
        if index % 10000 == 0:
            print(index)
    out_file.flush()
    out_file.close()
    return None


calculate_delivery_date()
