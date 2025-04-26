import pickle
import numpy as np

def show_feature_importance():
    # Load model and TF-IDF vectorizer
    with open('models/genre_classifier.pkl', 'rb') as f:
        model = pickle.load(f)

    tfidf = model.named_steps['tfidf']
    clf = model.named_steps['clf']

    feature_names = tfidf.get_feature_names_out()

    for genre_idx, genre_clf in enumerate(clf.estimators_):
        print(f"\n🎯 Top features for genre {genre_idx}:")
        
        # Get top features for this genre
        top_indices = np.argsort(genre_clf.coef_[0])[-10:]
        top_features = [feature_names[i] for i in top_indices]

        print("Top Words:", top_features)
