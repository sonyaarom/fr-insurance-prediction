import pandas as pd
import numpy as np


def create_stats_cols(data:pd.DataFrame, column: str) -> pd.DataFrame:
    stats = data.groupby(column).agg({
        'Exposure': 'sum',
        'VehPower': 'mean',
        'DrivAge': 'mean',
        'BonusMalus': 'mean'
    }).reset_index()

    data = data.merge(stats, on=column, how='left', suffixes=('', f'_{column}_Avg'))

    return data


def create_brand_frequency_bins(df, column='VehBrand', n_bins=2, labels = ['High', 'Medium', 'Low']):
    # Count the frequency of each brand
    brand_counts = df[column].value_counts()
    
    # Create bins based on frequency
    bins = pd.qcut(brand_counts, q=n_bins, labels=labels)
    
    # Create a mapping dictionary
    brand_frequency_map = dict(zip(brand_counts.index, bins))
    
    # Create new feature
    df[f'{column}_Frequency'] = df[column].map(brand_frequency_map)
    
    return df

#create a function to do log transformation of the columns and change the column name in the dataframe
def log_transform(data, column):
    data[f'{column}_log'] = np.log(data[column])
    return data