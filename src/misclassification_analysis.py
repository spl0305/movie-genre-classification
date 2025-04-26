import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(model, X_test, y_test, mlb):
    y_pred = model.predict(X_test)
    # Assuming y_test and y_pred are multi-label indicator arrays
    for i in range(y_test.shape[1]):  # Iterate through each genre
        genre_name = mlb.classes_[i]
        cm = confusion_matrix(y_test[:, i], y_pred[:, i])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix for {genre_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f'confusion_matrix_{genre_name}.png') # Save the plot
        plt.close() # Close the figure to free up resources
    print("Confusion matrices saved as PNG files.")
