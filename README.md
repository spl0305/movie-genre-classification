# movie-genre-classification
A well-trained model capable of accurately classifying movies into genres based on textual descriptions, along with insights into feature importance and misclassifications. 

🎬 Movie Genre Classification
Task: Predict the genre(s) of movies based on their textual descriptions using Machine Learning.
Assignment: GrowthLink Internship - Machine Learning Track

🧠 Problem Statement
Build a multi-label classification model that can predict one or more genres for a movie based on its plot summary.

📂 Project Structure

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

├── output_images.png
├── main.py
├── README.md
🛠 Approach

Step	Description
1. Load Data	Parsed train_data.txt (movie title, genre, description)
2. Preprocessing	Cleaned text: lowercasing, removing punctuation, stopwords
3. Feature Extraction	Used TF-IDF vectorization (max 5000 features)
4. Model Training	Trained One-vs-Rest Logistic Regression model
5. Evaluation	Calculated precision, recall, F1-score per genre
6. Feature Analysis	Identified top words per genre
7. Misclassification	Visualized confusion matrices for each genre

🔥 Technologies Used
Python 3
Scikit-learn
Pandas
NLTK
Matplotlib & Seaborn (for visualization)


📈 Results

Metric	Score
Micro F1-Score	48%
Macro F1-Score	16%
Samples F1-Score	35%

Model performs strongly on popular genres like drama, comedy, documentary.
Rare genres like musical, fantasy, war have lower performance (dataset imbalance).


📊 Insights

Top Features for Genres:

Example:
Comedy: funny, joke, humor, laugh
drama: family, love, relationship, father
Horror: murder, haunted, ghost, killer

Misclassifications Observed:
Thriller sometimes confused with Crime
Family movies occasionally mislabeled as Comedy

🚀 How to Run Locally
Clone the repository:
git clone https://github.com/your-username/movie-genre-classification.git
cd movie-genre-classification

Create a virtual environment and activate it:
python -m venv venv
.\venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Run the pipeline:
python main.py


👨‍💻 Author
SRI PUSHPA LATHA GARAGA
www.linkedin.com/in/sri-pushpa-latha-garaga


