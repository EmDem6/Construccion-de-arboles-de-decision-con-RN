import pandas as pd

import pandas as pd

# Read the .data file as a CSV
data_df = pd.read_csv('../../Datasets/CarEvaluation/car.data', header=None)

# Define the mapping for each categorical feature
buying_mapping = {'v-high': 4, 'high': 3, 'med': 2, 'low': 1}
maint_mapping = {'v-high': 4, 'high': 3, 'med': 2, 'low': 1}
doors_mapping = {'2': 2, '3': 3, '4': 4, '5-more': 5}
persons_mapping = {'2': 2, '4': 4, 'more': 6}
lug_boot_mapping = {'small': 1, 'med': 2, 'big': 3}
safety_mapping = {'low': 1, 'med': 2, 'high': 3}
class_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'v-good': 3}

# Apply the mappings to convert the categorical values to numerical labels
data_df[0] = data_df[0].map(buying_mapping)
data_df[1] = data_df[1].map(maint_mapping)
data_df[2] = data_df[2].map(doors_mapping)
data_df[3] = data_df[3].map(persons_mapping)
data_df[4] = data_df[4].map(lug_boot_mapping)
data_df[5] = data_df[5].map(safety_mapping)
data_df[6] = data_df[6].map(class_mapping)

data_df.dropna(inplace=True)

# Split the dataset into features (X) and labels (y)
X = data_df.values[:, :-1]
y = data_df.values[:, -1].astype('int')

