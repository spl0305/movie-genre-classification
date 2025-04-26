from src.train import train_model
from src.evaluate import evaluate_model

if __name__ == "__main__":
    model, X_test, y_test, mlb = train_model()
    evaluate_model(model, X_test, y_test, mlb)

from src.feature_importance import show_feature_importance

if __name__ == "__main__":
    model, X_test, y_test, mlb = train_model()
    evaluate_model(model, X_test, y_test, mlb)
    show_feature_importance()

from src.misclassification_analysis import plot_confusion_matrix

if __name__ == "__main__":
    model, X_test, y_test, mlb = train_model()
    evaluate_model(model, X_test, y_test, mlb)
    show_feature_importance()
    plot_confusion_matrix(model, X_test, y_test, mlb)



