from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import joblib
import os


def train_random_forest(X_train, y_train):
    
    rf = RandomForestClassifier(
        n_estimators=150,
        n_jobs=-1,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/random_forest.pkl")
    
    return rf

def train_xgboost(X_train, y_train):
    
    xgb = XGBClassifier(
        tree_method = "hist",
        n_estimators=200,
        
        max_depth=8,
        learning_rate=0.1,
        n_jobs=-1
    )
    
    xgb.fit(X_train, y_train)
    
    joblib.dump(xgb, "models/xgboost.pkl")
    
    
    return xgb