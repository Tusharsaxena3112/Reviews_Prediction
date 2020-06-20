import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


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

vectorizer = CountVectorizer()

reviews_text_final = vectorizer.fit_transform(
    reviews_text)  # bags of words (to convert text data into # numerical data)

train_reviews_text, test_reviews_text, train_reviews_sentiments, test_reviews_sentiments = train_test_split(
    reviews_text_final, review_sentiments, test_size=0.3, random_state=42)

# first model ie K -nn from sklearn

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(train_reviews_text, )

predictions_knn = knn.predict(test_reviews_text_final)

# dec = DecisionTreeClassifier(random_state=42)
#
# dec.fit(train_reviews_text_final,train_review_sentiments)
#
# dec.predict(test_reviews_text_final)
