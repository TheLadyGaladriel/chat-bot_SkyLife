import re
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

def preprocess_text(text):
    tokens = re.findall(r'\w+', text.lower())
    lemmas = [morph.parse(token)[0].normal_form for token in tokens]
    preprocessed_text = ' '.join(lemmas)
    return preprocessed_text

with open("intents(3).json", "r") as file:
    data = json.load(file)

X = []
y = []

for intent, intent_data in data.items():
    examples = intent_data["examples"]
    responses = intent_data["responses"]
    X.extend([preprocess_text(example) for example in examples])
    y.extend([intent] * len(examples))

df = pd.DataFrame({'Признаки': X, 'Метки классов': y})
my_tags = df['Метки классов'].unique()

vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), max_df=0.8)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

classifier = LinearSVC()
classifier.fit(X_train, y_train)

def classify_intent(text):
    preprocessed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([preprocessed_text])
    intent = classifier.predict(text_vectorized)[0]
    response = data[intent]["responses"][0]
    return intent, response


