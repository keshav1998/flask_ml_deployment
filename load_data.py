import pandas as pd
import gc


def create_feature_target():
    df = pd.read_csv('data/50_Startups.csv')
    X = df.iloc[:, :3]
    y = df.iloc[:, :-1]
    del df
    gc.collect()
    return X, y