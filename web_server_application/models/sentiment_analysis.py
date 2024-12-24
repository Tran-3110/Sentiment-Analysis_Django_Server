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
        # model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", cache_dir='model/model_sa_phobert')
        # model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")

        # Tải tokenizer (Xem BERT)
        # tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

        #Tiền xử lý
        sentence = self.__pre_processing_data(input_data['sentence'])
        # Tokenize sentence và chuyển thành tensor
        input_ids = torch.tensor([self.tokenizer.encode(sentence)])

        # Tiến hành triển khai model với input
        with torch.no_grad():
            out = self.model(input_ids)
            # Tính softmax để phân loại và in kết quả có dạng [[NEG, POS, NEU]]
            sentiment_result = out.logits.softmax(dim=-1).tolist()
            print(sentiment_result)
            processed_label = rating_sentiment(sentiment_result)
            return {'label': processed_label, 'sentence': input_data['sentence']}

    def perform_sentiment_analysis(self, input_data):
        # Nếu không có sentence
        if input_data['sentence'] is 'None': return None
        # Nếu đã có sentiment và sentence (Không cần thực hiện nữa)
        if input_data['sentiment'] != 'None' and input_data['sentence'] != 'None': return input_data
        return self.__perform_model(input_data)


def rating_sentiment(sentiment_result, delta=0.15):
    sentiment_output = sentiment_result[0]
    label = ['NEG', 'POS', 'NEU']

    # Gán nhãn với từng tỉ lệ xuất hiện của từng loại cảm xúc
    sentiment_rating = [{'label': x, 'rate': sentiment_output[label.index(x)] / sum(sentiment_output)} for x in label]
    sentiment_rating.sort(key=lambda x: x['rate'], reverse=True)

    max_rating = sentiment_rating[0]
    mid_rating = sentiment_rating[1]

    if abs(max_rating['rate'] - mid_rating['rate']) > delta:
        return 'NEU' if max_rating['label'] == 'NEU' else max_rating['label']
    else:
        if max_rating['label'] != 'NEU' and mid_rating['label'] != 'NEU':
            return 'NEU'
        else:
            return max_rating['label']
    # return max(sentiment_rating, key=lambda x: x['rate'])['label']

# Test model
# if __name__ == '__main__':
#     examples = [
#         {'sentiment': None, 'sentence': 'Nhà này cũng đẹp nhưng tôi không thích nó lắm'},
#         {'sentiment': 'NEG', 'sentence': 'Chỗ này thật rác rưởi'},
#         {'sentiment': None, 'sentence': 'Dịch vụ xung quanh rất ổn và tôi cực kì thích nó'},
#         {'sentiment': 'POS', 'sentence': 'Nhà này cũng đẹp nhưng tôi không thích nó lắm'},
#         {'sentiment': None, 'sentence': 'Bữa sáng hơi dở nhưng chất lượng phòng xịn lắm'},
#         {'sentiment': None, 'sentence': 'Netflix hết hạn sử dụng nên tôi không thể xem phim được'},
#     ]
#     sentiment_analysis = SentimentAnalysis()
#     for example in examples:
#         result = sentiment_analysis.perform_sentiment_analysis(example)
#         print(result.keys())
#         # print("\n")

# def pre_processing_data(sentence):
#     # Load VnCoreNLP
#     segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.getenv('VN_CORE_NLP_PATH'))
#     output = segmenter.word_segment(sentence)  # Xử lý văn bản
#     # Join các từ trong list để merge lại
#     return ''.join([word for word in output])
#
#
# def perform_model(sentence):
#     # Tải model
#     # Khai báo option cache_dir dùng để lưu model vào thư mục chỉ định (Nặng 500mb 🥲😃). Nếu không thì sẽ lưu vào cache
#     # model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", cache_dir='model/model_sa_phobert')
#
#     model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
#
#     # Tải tokenizer (Xem BERT)
#     tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)
#
#     # Tokenize sentence và chuyển thành tensor
#     input_ids = torch.tensor([tokenizer.encode(sentence)])
#
#     # Tiến hành triển khai model với input
#     with torch.no_grad():
#         out = model(input_ids)
#         # Tính softmax để phân loại và in kết quả có dạng [[NEG, POS, NEU]]
#         sentiment_result = out.logits.softmax(dim=-1).tolist()
#         print(sentiment_result)
#         processed_label = rating_sentiment(sentiment_result)
#         return {'label': processed_label, 'sentence': sentence}


# # Đưa vào một đối tượng gồm sentiment: kết quả thực hiện và sentence: câu tương ứng để thực hiện
# def perform_sentiment_analysis(input_data):
#     # Nếu không có sentence
#     if input_data['sentence'] is None: return None
#     # Nếu đã có sentiment và sentence (Không cần thực hiện nữa)
#     if input_data['sentiment'] is not None and input_data['sentence'] is not None: return input_data
#     pre_process_sentence = pre_processing_data(input_data['sentence'])
#     return perform_model(pre_process_sentence)


# if __name__ == '__main__':
#     example = {'sentiment': None, 'sentence': 'Nhà này rất ổn, dịch vụ rất chất lượng nhưng bữa sáng hơi tệ'}
#     result = perform_sentiment_analysis(example)
#     print(result)
