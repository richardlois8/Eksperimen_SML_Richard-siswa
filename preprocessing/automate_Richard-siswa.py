import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df = df.drop_duplicates()
    df['screen_area'] = df['px_height'] * df['px_width']
    
    features = df.drop('price_range', axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
    features_scaled_df['price_range'] = df['price_range'].values
    return features_scaled_df

def save_processed(df, out_path='mobile_price_classification_preprocessing.csv'):
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1]
    df = load_data(filepath)
    df_processed = preprocess_data(df)
    save_processed(df_processed)
    print("Preprocessing done! Preprocessed data saved as mobile_price_classification_preprocessing.csv")
