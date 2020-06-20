class Review:
    def __init__(self,review_text,rate):
        self.review_text = review_text
        self.rate = rate
        self.sentiment = self.get_sentiment(rate)

    def get_sentiment(self,rate):
        if rate<=2:
            return 'NEGATIVE'
        elif rate == 3:
            return 'NEUTRAL'
        else:
            return 'POSITIVE'



with open('data_reviews') as f:
    file = f.readlines()
    for i in file:

