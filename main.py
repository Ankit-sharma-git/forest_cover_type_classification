from src.data_loader import load_data

from src.preprocessing import preprocessing_data
from src.train import train_random_forest, train_xgboost
from src.evaluate import evaluate_model
from src.visualize import plot_confusion_matrix, plot_feature_importance

def main():
    
    df = load_data()
    
    X_train, X_test, y_train, y_test, encoder = preprocessing_data(df)
    
    rf = train_random_forest(X_train, y_train)
    
    preds_rf = evaluate_model(rf, X_test, y_test)
    
    plot_confusion_matrix(rf, X_test, y_test)
    
    plot_feature_importance(rf , X_train.columns)
    print("Unique y_train", sorted(set(y_train)))
    xgb = train_xgboost(X_train, y_train)
    
    preds_xgb = evaluate_model(xgb, X_test, y_test)
    
    plot_confusion_matrix(xgb, X_test, y_test)
    plot_feature_importance(xgb, X_train.columns)
    
if __name__ == "__main__":
    main()
    