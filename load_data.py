import pandas as pd
import gc


def create_feature_target():
    df = pd.read_csv('data/50_Startups.csv')
    X = df[['R&D Spend','Administration','Marketing Spend']]
    y = df['Profit']
    del df
    gc.collect()
    return X, y