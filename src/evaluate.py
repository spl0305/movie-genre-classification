from sklearn.metrics import classification_report
import pickle

def evaluate_model(model, X_test, y_test, mlb):
    print("📊 Evaluation Report:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))
