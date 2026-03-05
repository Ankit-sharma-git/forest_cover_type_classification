from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

def preprocessing_data(df):
    
    X = df.drop("Cover_Type", axis=1)
    y = df["Cover_Type"]
    
    encoder = LabelEncoder()
    y  = encoder.fit_transform(y)
    
    
    X_train,  X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y,
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test, encoder