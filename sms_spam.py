import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


file_path = r"A:\spam.csv"


df = pd.read_csv(file_path, encoding='latin-1')

 
df = df[['v1', 'v2']]
df.columns = ['label', 'message']


df['label'] = df['label'].map({'ham': 0, 'spam': 1})


tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['message'])
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\n" + "="*30)
print(" SMS SPAM CHECKER ")
print("="*30)

user_msg = input("Enter the SMS text: ")

if user_msg.strip():
    user_data = tfidf.transform([user_msg])
    prediction = model.predict(user_data)
    
    if prediction[0] == 1:
        print("RESULT: This is SPAM!")
    else:
        print("RESULT: This is a LEGITIMATE message.")