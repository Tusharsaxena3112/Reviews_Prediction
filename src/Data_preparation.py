import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


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

train_reviews, test_reviews = train_test_split(reviews, test_size=0.3, random_state=42)

train_reviews_text = [x.review_text for x in train_reviews]
train_review_sentiments = [x.sentiment for x in train_reviews]

test_reviews_text = [x.review_text for x in test_reviews]
test_reviews_sentiments = [x.sentiment for x in test_reviews]

vectorizer = CountVectorizer()

test_reviews_text_final = vectorizer.fit_transform(
    train_reviews_text)  # bags of words (to convert text data into # numerical data)
