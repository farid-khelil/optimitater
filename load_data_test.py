import pandas as pd
import numpy as np


df = pd.read_csv('/home/farid/pfe/data/processed/ransomware/RISS.csv')
df[df.columns[-1]] = df[df.columns[-1]].round().astype(int)
print(len(df))
print(len(df))
bingin = df[df[df.columns[-1]] == 0]
print(len(bingin))
ransomweare = df[df[df.columns[-1]] == 1]
print(len(ransomweare))