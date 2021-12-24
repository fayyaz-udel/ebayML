import numpy as np
import pandas as pd

df = pd.read_csv("./data/train.tsv", sep="\t")

df['category_id'].value_counts()
print("sdfsd")