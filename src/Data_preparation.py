import json
import random

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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


def evenly_distribute(reviews):
    positive_reviews = list(filter(lambda x: x.sentiment == 'POSITIVE', reviews))
    negative_reviews = list(filter(lambda x: x.sentiment == 'NEGATIVE', reviews))
    reviews = positive_reviews[:len(negative_reviews)] + negative_reviews
    random.shuffle(reviews)
    return reviews


reviews = []
with open('Books_small_10000.json') as f:
    for line in f:
        l = json.loads(line)
        reviews.append(Review(l['reviewText'], l['overall']))

reviews = evenly_distribute(reviews)
reviews_text = [x.review_text for x in reviews]
review_sentiments = [x.sentiment for x in reviews]

# splitting data into train and test
train_reviews_text, test_reviews_text, train_reviews_sentiments, test_reviews_sentiments = train_test_split(
    reviews_text, review_sentiments, test_size=0.2, random_state=42)

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

# SVM model
svm = SVC(kernel='linear', random_state=42)
svm.fit(train_reviews_text_vector, train_reviews_sentiments)
prediction_svm = svm.predict(test_reviews_text_vector)
accuracy_score_svm = accuracy_score(np.array(test_reviews_sentiments), prediction_svm)


# Evaluation of the models

f1_score_knn = f1_score(test_reviews_sentiments, predictions_knn, average=None, labels=['POSITIVE', 'NEGATIVE'])
print(f1_score_knn)

f1_score_dec = f1_score(test_reviews_sentiments, predication_dec, labels=['POSITIVE', 'NEGATIVE'], average=None)
print(f1_score_dec)

f1_score_svm = f1_score(test_reviews_sentiments, prediction_svm, average=None, labels=['POSITIVE', 'NEGATIVE'])
print(f1_score_svm)

test_data = ['Bad idea']
test_data_vector = vectorizer.transform(test_data)
print(svm.predict(test_data_vector))
