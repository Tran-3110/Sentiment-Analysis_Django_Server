import os
import pickle

from web_server_application.models.sentiment_analysis import SentimentAnalysis

with open(os.getenv('SPAM_FILTER_PATH'), 'rb') as f:
    model = pickle.load(f)


def api_process(input_data):
    if perform_spam_filter(input_data['sentence']):
        return {
            'result': True,
            'content': perform_sentiment_analysis(input_data)
        }
    else:
        return {
            'result': False,
            'content': None
        }


def perform_spam_filter(message, delta=0):
    result = model.predict_proba([message])[0]
    print(result)
    if abs(result[0] - result[1]) == delta:  # Delta = 0 vì trọng số lúc này là bằng nhau
        return False  # Nếu chênh lệch quá bé thì cho là spam
    else:
        return result[0] > result[1]  # True khi không spam, False khi spam


def perform_sentiment_analysis(input_data):
    return SentimentAnalysis().perform_sentiment_analysis(input_data)
