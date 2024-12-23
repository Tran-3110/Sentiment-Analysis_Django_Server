# serializers.py
from rest_framework import serializers

class SentimentAnalysisSerializer(serializers.Serializer):
    sentence = serializers.CharField()  # Câu văn đầu vào
    sentiment = serializers.CharField(max_length=10, required=False)  # Kết quả phân tích cảm xúc (Có thể bỏ qua khi gửi yêu cầu)
