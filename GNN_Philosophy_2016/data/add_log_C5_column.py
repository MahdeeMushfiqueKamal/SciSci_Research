import pandas as pd
import numpy as np

files = [
    "PhilosophyPapers_2000_15.csv",
    "PhilosophyPapers_2016_test.csv",
    "PhilosophyPapers_2016_validation.csv"
]

for file in files:
    df = pd.read_csv(file)
    df["C5_log"] = np.log1p(df["C5"])
    df.to_csv(file, index=False)