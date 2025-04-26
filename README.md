# 🎬 Movie Genre Classification

A Machine Learning project to predict movie genres based on textual movie descriptions.

> Developed as part of the **GrowthLink Internship** Machine Learning assignment.

---

## 🧠 Problem Statement

Build a multi-label classification model that can accurately predict one or more genres for a movie based on its plot summary.

---

## 📂 Project Structure

```plaintext
movie-genre-classification/
├── data/
│   └── train_data.txt
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── feature_importance.py
│   └── misclassification_analysis.py
├── models/
│   ├── genre_classifier.pkl
│   └── genre_binarizer.pkl
├── outputs/
│   ├── confusion_matrix_action.png
│   ├── confusion_matrix_comedy.png
│   └── ... (confusion matrices for each genre)
├── requirements.txt
├── main.py
├── README.md

🛠 Approach

Step	Description

📥 Load Data	Parsed text file into movie descriptions and genres
🧹 Preprocessing	Cleaned text (lowercase, punctuation removal, stopwords removal)
🔢 Feature Extraction	Applied TF-IDF vectorization (max 5000 features)
🤖 Model Training	Trained OneVsRestClassifier with Logistic Regression
📊 Evaluation	Measured precision, recall, F1-score
📈 Feature Analysis	Identified important words influencing genre predictions
📉 Misclassification Analysis	Plotted confusion matrices for each genre

🚀 Technologies Used

Python 3.11
Scikit-learn
Pandas
NLTK
Matplotlib
Seaborn

📈 Model Performance

Metric	Score
Micro F1-Score	48%
Macro F1-Score	16%
Samples F1-Score	35%

✅ The model performs strongly for popular genres like drama, comedy, and documentary.

🔥 Feature Importance (Sample)

Comedy: funny, joke, humor, laugh, hilarious
Horror: murder, haunted, ghost, killing
Drama: family, father, love, relationship

📉 Misclassification Insights

Thriller is often confused with Crime.
Family movies sometimes predicted as Comedy.
Rare genres (like musical, war) have lower accuracy.

Confusion matrices are available as in png format for each genre

📋 How to Run Locally
1.Clone the repository:
 git clone https://github.com/your-username/movie-genre-classification.git
 cd movie-genre-classification
2.Create a virtual environment:
 python -m venv venv
.\venv\Scripts\activate
3.Install required packages:
 pip install -r requirements.txt
4.Run the project:
 python main.py

📎 Requirements
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn

📌 Future Improvements 
Use pre-trained word embeddings (Word2Vec, GloVe)
Try deep learning models (LSTM, Transformers)
Handle rare genres with data augmentation or SMOTE
Perform hyperparameter tuning




👨‍💻 Author
SRI PUSHPA LATHA GARAGA
www.linkedin.com/in/sri-pushpa-latha-garaga


