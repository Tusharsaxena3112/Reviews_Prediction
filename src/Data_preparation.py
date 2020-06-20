import json

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class Review:
    def __init__(self, review_text, rate):
        self.review_text = review_text
        self.rate = rate
        self.sentiment = self.get_sentiment(rate)

    def get_sentiment(self, rate):
        if rate <= 2:
            return 'NEGATIVE'
        elif rate == 3:
            return 'NEUTRAL'
        else:
            return 'POSITIVE'


reviews = []
with open('Books_small.json') as f:
    for line in f:
        l = json.loads(line)
        reviews.append(Review(l['reviewText'], l['overall']))

reviews_text = [x.review_text for x in reviews]
review_sentiments = [x.sentiment for x in reviews]

# splitting data into train and test
train_reviews_text, test_reviews_text, train_reviews_sentiments, test_reviews_sentiments = train_test_split(
    reviews_text, review_sentiments, test_size=0.3, random_state=42)

vectorizer = CountVectorizer()

train_reviews_text_vector = vectorizer.fit_transform(
    train_reviews_text)  # bags of words (to convert text data into # numerical data)

test_reviews_text_vector = vectorizer.transform(test_reviews_text)

# first model ie K -nn from sklearn

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(train_reviews_text_vector, train_reviews_sentiments)
predictions_knn = knn.predict(test_reviews_text_vector)
accuracy_score_knn = accuracy_score(np.array(test_reviews_sentiments), predictions_knn)

# Another model ie Decision Tree

dec = DecisionTreeClassifier(random_state=42)
dec.fit(train_reviews_text_vector, train_reviews_sentiments)
predication_dec = dec.predict(test_reviews_text_vector)
accuracy_score_dec = accuracy_score(np.array(test_reviews_sentiments), predication_dec)

# Evaluation of the models

from sklearn.metrics import f1_score

f1_score_knn = f1_score(test_reviews_sentiments, predictions_knn, average=None,
                        labels=['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
print(f1_score_knn)

f1_score_dec = f1_score(test_reviews_sentiments, predication_dec, labels=['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
                        average=None)
print(f1_score_dec)
