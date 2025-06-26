import numpy as np

def create_simple_features(df, target_col='Weekly_Sales'):
    features_df = df.copy()
    for lag in [1,2,3,4]:
        features_df[f'sales_lag_{lag}'] = features_df[target_col].shift(lag)
    features_df['sales_avg_4weeks'] = features_df[target_col].rolling(4).mean()
    features_df['sales_avg_8weeks'] = features_df[target_col].rolling(8).mean()
    features_df['sales_avg_12weeks'] = features_df[target_col].rolling(12).mean()
    features_df['month'] = features_df.index.month
    features_df['quarter'] = features_df.index.quarter
    features_df['week_of_year'] = features_df.index.isocalendar().week
    features_df['is_holiday_season'] = features_df['month'].isin([11,12,1]).astype(int)
    features_df['is_back_to_school'] = features_df['month'].isin([8,9]).astype(int)
    features_df['trend'] = np.arange(len(features_df))
    return features_df
