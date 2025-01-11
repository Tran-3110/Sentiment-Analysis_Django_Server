import dotenv, os
import torch, py_vncorenlp
from transformers import RobertaForSequenceClassification, AutoTokenizer

dotenv.load_dotenv()

# Singleton class
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# Implement Singleton bằng cách dùng metaclass
class SentimentAnalysis(metaclass=Singleton):
    def __init__(self):
        self.segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.getenv('VN_CORE_NLP_PATH'))
        self.model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
        self.tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

    def __pre_processing_data(self, sentence):
        output = self.segmenter.word_segment(sentence)  # Xử lý văn bản
        # Join các từ trong list để merge lại
        return ''.join([word for word in output])

    def __perform_model(self, input_data):
        # Tải model
        # Khai báo option cache_dir dùng để lưu model vào thư mục chỉ định (Nặng 500mb 🥲😃). Nếu không thì sẽ lưu vào cache

        # Tải tokenizer (Xem BERT)
        # tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

        #Tiền xử lý
        sentence = self.__pre_processing_data(input_data['sentence'])
        # Tokenize sentence và chuyển thành tensor
        input_ids = torch.tensor([self.tokenizer.encode(sentence)])

        # Tiến hành triển khai model với input
        with torch.no_grad():
            out = self.model(input_ids)
            # Tính softmax để phân loại và in kết quả có dạng [[negative, positive, neutral]]
            sentiment_result = out.logits.softmax(dim=-1).tolist()
            print(sentiment_result)
            processed_label = rating_sentiment(sentiment_result)
            return {'label': processed_label, 'sentence': input_data['sentence']}

    def perform_sentiment_analysis(self, input_data):
        # Nếu không có sentence
        if input_data['sentence'] == 'None': return None
        # Nếu đã có sentiment và sentence (Không cần thực hiện nữa)
        if input_data['sentiment'] != 'None' and input_data['sentence'] != 'None': return input_data
        return self.__perform_model(input_data)


def rating_sentiment(sentiment_result):
    sentiment_output = sentiment_result[0]
    label = ['negative', 'positive', 'neutral']

    # Gán nhãn với từng tỉ lệ xuất hiện của từng loại cảm xúc
    sentiment_rating = [{'label': x, 'rate': sentiment_output[label.index(x)] / sum(sentiment_output)} for x in label]
    sentiment_rating.sort(key=lambda x: x['rate'], reverse=True)

    max_rating = sentiment_rating[0]
    mid_rating = sentiment_rating[1]

    # Bỏ phần ngưỡng xác định với các case gần nhau, cho trả về trực tiếp label có tỉ lệ cao nhất
    # if abs(max_rating['rate'] - mid_rating['rate']) > delta:
    #     return 'neutral' if max_rating['label'] == 'neutral' else max_rating['label']
    # else:
    #     if max_rating['label'] != 'neutral' and mid_rating['label'] != 'neutral':
    #         return 'neutral'
    #     else:
    #         return max_rating['label']
    return max_rating['label']
