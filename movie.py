import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


path = r"A:\Genre Classification Dataset\train_data.txt"


df = pd.read_csv(path, sep=' ::: ', engine='python', names=['ID', 'Title', 'Genre', 'Plot'])
df['Plot'] = df['Plot'].fillna('')


tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['Plot'])
y = df['Genre']


model = LogisticRegression(max_iter=1000)
model.fit(X, y)


print("\n--- Movie Genre Predictor ---")
user_plot = input("Enter a movie plot summary: ")


if user_plot.strip():
    prediction = model.predict(tfidf.transform([user_plot]))
    print(f"\nPredicted Genre: {prediction[0]}")
else:
    print("You didn't enter a plot!")