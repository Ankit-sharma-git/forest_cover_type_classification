import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np


def plot_confusion_matrix(model, X_test, y_test):

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

    plt.title("Confusion Matrix")

    plt.show()


def plot_feature_importance(model, feature_names):

    if not hasattr(model, "feature_importances_"):

        print("Model does not support feature importance. ")

        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices])

    plt.yticks(range(len(indices)), feature_names[indices])

    plt.title("Top 10 Important Features")

    plt.show()
