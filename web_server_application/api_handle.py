import pickle

from web_server_application.models.sentiment_analysis import SentimentAnalysis

with open('web_server_application/lib/models/spamfilter_model.pkl', 'rb') as f:
    model = pickle.load(f)

def api_process(input_data):
    if perform_spam_filter(input_data['sentence']):
        return {
            'result': True,
            'content': perform_sentiment_analysis(input_data)
        }
    else: return {
        'result': False,
        'content': None
    }

def perform_spam_filter(message, delta=0.15):
    result = model.predict_proba([message])[0]
    if abs(result[0] - result[1]) <= delta:
        return True  # Nếu chênh lệch quá bé thì cho là không phải spam
    else:
        return result[0] > result[1]  # True khi không spam, False khi spam


def perform_sentiment_analysis(input_data):
    return SentimentAnalysis().perform_sentiment_analysis(input_data)