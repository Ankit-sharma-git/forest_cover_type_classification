from sklearn.datasets import fetch_covtype
import pandas as pd

def load_data():
    data = fetch_covtype(as_frame= True)
    df = data.frame
    
    print("Dataset Shape:", df.shape)
    
    return df