class Review:
    def __init__(self,review_text,rate):
        self.review_text = review_text
        self.rate = rate
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        

with open('data_reviews') as f:
    file = f.readlines()
    for i in file:

