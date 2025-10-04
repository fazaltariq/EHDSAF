# Re-import necessary libraries and load the dataset
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv('floods.csv', encoding='ISO-8859-1')

# Display first few rows
print("First few rows of the dataset:")
print(df.head())