import csv
from datetime import datetime as hamed
import datetime

import pandas as pd


def date_by_adding_business_days(from_date, add_days):
    business_days_to_add = add_days
    current_date = from_date
    while business_days_to_add > 0:
        current_date += datetime.timedelta(days=1)
        weekday = current_date.weekday()
        if weekday >= 5:  # sunday = 6
            continue
        business_days_to_add -= 1
    return current_date


df = pd.read_csv("./data/quiz.tsv", sep="\t")
with open('records.tsv', 'w+', newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    counter = 0
    df['declared_handling_days'].fillna(0, inplace=True)
    for index, row in df.iterrows():
        counter += 1
        print(str(counter)+"   " + str(row['declared_handling_days']) + "   " + str(row['carrier_min_estimate']))
        t = hamed.strptime(row['payment_datetime'][:10], '%Y-%m-%d')
        delta = int(row['declared_handling_days']) + int(row['carrier_min_estimate'])
        writer.writerow([row['record_number'], str(date_by_adding_business_days(t, delta))[:10]])

    tsvfile.flush()
    tsvfile.close()
