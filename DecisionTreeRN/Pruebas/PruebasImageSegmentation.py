import pandas as pd

# Read the .data file to extract the actual data instances and labels
data_df = pd.read_csv('../../Datasets/habermanSurvival/haberman.data', header=None)

# Split the dataset into features (X) and labels (y)
X = data_df.values[:,0:-1]
y = data_df.values[:,-1]
