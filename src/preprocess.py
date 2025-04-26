import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def load_txt_dataset(file_path):
    ids, titles, genres, descriptions = [], [], [], []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                id_, title, genre, desc = parts
                ids.append(id_)
                titles.append(title.strip())
                genres.append([g.strip().lower() for g in genre.strip().split(',')])
                descriptions.append(desc.strip())

    df = pd.DataFrame({
        'id': ids,
        'title': titles,
        'genre': genres,
        'description': descriptions
    })

    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    stop_words = set(stopwords.words("english"))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_data(df):
    df['clean_description'] = df['description'].apply(clean_text)
    return df
