import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df = df.drop_duplicates()
    df['screen_area'] = df['px_height'] * df['px_width']
    X = df.drop('price_range', axis=1)
    y = df['price_range']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def save_processed(X, y, out_X='X_preprocessing.csv', out_y='y_preprocessing.csv'):
    pd.DataFrame(X).to_csv(out_X, index=False)
    y.to_csv(out_y, index=False)

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] 
    df = load_data(filepath)
    X, y = preprocess_data(df)
    save_processed(pd.DataFrame(X), y)
    print("Preprocessing done! Processed data saved.")
